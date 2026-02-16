"""Core sweep loop: extract, train, MC-eval across a hyperparameter grid."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.data.truthfulqa import TruthfulQADatasetManager
from src.evaluation.truthfulqa import evaluate_multiple_choice
from src.models.loader import LoadedModel, load_causal_model
from src.stages.common import set_random_seeds, setup_environment
from src.steering.extract import ActivationExtractor, compute_caa_vector
from src.steering.mlp import SteeringMLP
from src.steering.training import (
    GenTrainingConfig,
    MCTrainingConfig,
    train_gen_mlp,
    train_mc_mlp,
)
from src.steering.vector_bank import VectorBank, VectorBankBuilder
from src.sweep.config import SweepConfig, combo_dir_name

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class SweepCheckpoint:
    """Resumable progress tracking for sweep runs."""

    def __init__(self, sweep_dir: Path) -> None:
        self.path = sweep_dir / "sweep_progress.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"extracted_layers": [], "completed_combos": [], "results": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def is_layer_extracted(self, layer: int) -> bool:
        return layer in self._data["extracted_layers"]

    def mark_layer_extracted(self, layer: int) -> None:
        if layer not in self._data["extracted_layers"]:
            self._data["extracted_layers"].append(layer)
            self._save()

    def is_combo_complete(self, layer: int, combo_id: str) -> bool:
        return f"{layer}/{combo_id}" in self._data["completed_combos"]

    def mark_combo_complete(
        self, layer: int, combo_id: str, result: dict
    ) -> None:
        key = f"{layer}/{combo_id}"
        if key not in self._data["completed_combos"]:
            self._data["completed_combos"].append(key)
        self._data["results"][key] = result
        self._save()

    def get_all_results(self) -> Dict[str, dict]:
        return self._data.get("results", {})


# ---------------------------------------------------------------------------
# Extraction (once per layer)
# ---------------------------------------------------------------------------

def run_layer_extraction(
    loaded: LoadedModel,
    dataset: TruthfulQADatasetManager,
    steering_pool: List[int],
    layer: int,
    sweep_dir: Path,
    config: dict,
) -> VectorBank:
    """Extract CAA vectors for a single layer."""
    layer_dir = sweep_dir / "vectors" / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    steering_cfg = config.get("steering", {})
    seed = config.get("run", {}).get("seed", 42)

    extractor = ActivationExtractor(
        loaded,
        layer,
        max_length=steering_cfg.get("max_length", 512),
        batch_size=steering_cfg.get("extract_batch_size", 8),
        safe_attention=steering_cfg.get("safe_attention", False),
    )

    pool_pos, pool_neg, _valid_prompt_indices = dataset.build_caa_prompts(
        steering_pool
    )
    LOG.info("Layer %d: extracting from %d prompt pairs", layer, len(pool_pos))

    pos_acts, pos_valid = extractor.collect_mean_activations(pool_pos)
    neg_acts, neg_valid = extractor.collect_mean_activations(pool_neg)

    valid_pairs = sorted(set(pos_valid) & set(neg_valid))
    pos_mask = torch.tensor([i in valid_pairs for i in pos_valid])
    neg_mask = torch.tensor([i in valid_pairs for i in neg_valid])
    pos_acts, neg_acts = pos_acts[pos_mask], neg_acts[neg_mask]

    if len(pos_acts) == 0:
        raise RuntimeError(
            f"No valid activation pairs at layer {layer}"
        )

    normalize = steering_cfg.get("normalize_vector", False)
    base_vector = compute_caa_vector(pos_acts, neg_acts, normalize=normalize)

    builder = VectorBankBuilder(
        pos_acts, neg_acts, normalize=normalize, seed=seed
    )
    bank_cfg = steering_cfg.get("vector_bank", {})
    vector_bank = builder.build(
        num_vectors=bank_cfg.get("num_vectors", 12),
        sample_size_range=(
            bank_cfg.get("min_samples", 30),
            bank_cfg.get("max_samples", 50),
        ),
    )

    torch.save(base_vector.cpu(), layer_dir / "base_vector.pt")
    torch.save(
        {
            "base_vector": vector_bank.base_vector.cpu(),
            "vectors": [v.cpu() for v in vector_bank.vectors],
            "indices": vector_bank.indices,
        },
        layer_dir / "vector_bank.pt",
    )

    LOG.info(
        "Layer %d extraction done: %d pairs, dim=%d, norm=%.4f",
        layer, len(pos_acts), pos_acts.shape[1], base_vector.norm().item(),
    )
    return vector_bank


def _load_vector_bank(sweep_dir: Path, layer: int) -> VectorBank:
    vb_path = sweep_dir / "vectors" / f"layer_{layer:02d}" / "vector_bank.pt"
    data = torch.load(vb_path, weights_only=False)
    return VectorBank(
        base_vector=data["base_vector"],
        vectors=data["vectors"],
        indices=data["indices"],
    )


# ---------------------------------------------------------------------------
# Single HP combo: train + MC-eval
# ---------------------------------------------------------------------------

def run_hp_combo(
    loaded: LoadedModel,
    dataset: TruthfulQADatasetManager,
    train_indices: List[int],
    val_items: List[dict],
    vector_bank: VectorBank,
    layer: int,
    lr: float,
    mse_reg: float,
    sweep_dir: Path,
    config: dict,
) -> Dict[str, Any]:
    """Train MC+Gen MLPs with specific HPs, then run MC evaluation on val set."""
    cid = combo_dir_name(lr, mse_reg)
    combo_dir = sweep_dir / "results" / f"layer_{layer:02d}" / cid
    combo_dir.mkdir(parents=True, exist_ok=True)

    device = loaded.primary_device
    param_dtype = next(loaded.model.parameters()).dtype
    hidden_dim = vector_bank.base_vector.shape[0]

    mlp_cfg = config.get("mlp", {})
    arch_cfg = mlp_cfg.get("architecture", {})
    steering_cfg = config.get("steering", {})
    seed = config.get("run", {}).get("seed", 42)
    max_length = steering_cfg.get("max_length", 512)

    result: Dict[str, Any] = {
        "layer": layer,
        "lr": lr,
        "mse_reg": mse_reg,
        "combo_id": cid,
    }

    # ---- Train MC MLP ----
    mlp_mc = SteeringMLP(
        input_dim=hidden_dim,
        hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
        dropout=arch_cfg.get("dropout", 0.1),
        bottleneck_dim=arch_cfg.get("bottleneck_dim"),
    ).to(device, dtype=param_dtype)

    base_mc_cfg = mlp_cfg.get("mc_training", {})
    mc_cfg = MCTrainingConfig(
        epochs=base_mc_cfg.get("epochs", 1),
        steps_per_epoch=base_mc_cfg.get("steps_per_epoch", 50),
        batch_size=base_mc_cfg.get("batch_size", 8),
        margin=base_mc_cfg.get("margin", 1.0),
        lr=lr,
        weight_decay=base_mc_cfg.get("weight_decay", 0.0),
        grad_clip=base_mc_cfg.get("grad_clip", 1.0),
        mse_reg=mse_reg,
        gradient_accumulation_steps=base_mc_cfg.get(
            "gradient_accumulation_steps", 1
        ),
    )

    mc_history = train_mc_mlp(
        mlp_mc,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        dataset=dataset,
        train_indices=train_indices,
        vector_bank=vector_bank,
        layer_index=layer,
        primary_device=device,
        max_length=max_length,
        config=mc_cfg,
        seed=seed + 1,
    )

    mc_valid = bool(mc_history.get("loss")) and all(
        v == v for v in mc_history["loss"]  # NaN check
    )
    if mc_valid:
        torch.save(mlp_mc.state_dict(), combo_dir / "mlp_mc_state_dict.pt")
    result["mc_train_valid"] = mc_valid
    result["mc_final_loss"] = (
        mc_history["loss"][-1] if mc_history.get("loss") else None
    )
    result["mc_final_acc"] = (
        mc_history["accuracy"][-1] if mc_history.get("accuracy") else None
    )

    # ---- Train Gen MLP ----
    mlp_gen = SteeringMLP(
        input_dim=hidden_dim,
        hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
        dropout=arch_cfg.get("dropout", 0.1),
        bottleneck_dim=arch_cfg.get("bottleneck_dim"),
    ).to(device, dtype=param_dtype)

    base_gen_cfg = mlp_cfg.get("gen_training", {})
    gen_cfg = GenTrainingConfig(
        epochs=base_gen_cfg.get("epochs", 1),
        steps_per_epoch=base_gen_cfg.get("steps_per_epoch", 40),
        batch_size=base_gen_cfg.get("batch_size", 4),
        lr=lr,
        weight_decay=base_gen_cfg.get("weight_decay", 0.0),
        grad_clip=base_gen_cfg.get("grad_clip", 1.0),
        mse_reg=mse_reg,
        gradient_accumulation_steps=base_gen_cfg.get(
            "gradient_accumulation_steps", 1
        ),
    )

    gen_history = train_gen_mlp(
        mlp_gen,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        dataset=dataset,
        train_indices=train_indices,
        vector_bank=vector_bank,
        layer_index=layer,
        primary_device=device,
        max_length=max_length,
        config=gen_cfg,
        seed=seed + 2,
    )

    gen_valid = bool(gen_history.get("loss")) and all(
        v == v for v in gen_history["loss"]
    )
    if gen_valid:
        torch.save(mlp_gen.state_dict(), combo_dir / "mlp_gen_state_dict.pt")
    result["gen_train_valid"] = gen_valid
    result["gen_final_loss"] = (
        gen_history["loss"][-1] if gen_history.get("loss") else None
    )

    # Save training history
    with open(combo_dir / "training_history.json", "w") as f:
        json.dump({"mc": mc_history, "gen": gen_history}, f, indent=2)

    # ---- MC Evaluation (Phase 1 proxy) ----
    base_vec = vector_bank.base_vector.to(device, dtype=param_dtype)
    mc_eval: Dict[str, Any] = {}

    if mc_valid:
        mlp_mc.eval()
        with torch.no_grad():
            mc_vector = mlp_mc(base_vec.unsqueeze(0)).squeeze(0)
        mc_result = evaluate_multiple_choice(
            loaded.model,
            loaded.tokenizer,
            val_items,
            layer_index=layer,
            steering_vector=mc_vector,
            scale=1.0,
            primary_device=device,
            seed=seed,
        )
        mc_eval["mlp_mc"] = {
            "accuracy": mc_result["stats"].accuracy,
            "avg_correct_prob": mc_result["stats"].avg_correct_prob,
            "total": mc_result["stats"].total,
        }

    if gen_valid:
        mlp_gen.eval()
        with torch.no_grad():
            gen_vector = mlp_gen(base_vec.unsqueeze(0)).squeeze(0)
        gen_mc_result = evaluate_multiple_choice(
            loaded.model,
            loaded.tokenizer,
            val_items,
            layer_index=layer,
            steering_vector=gen_vector,
            scale=1.0,
            primary_device=device,
            seed=seed,
        )
        mc_eval["mlp_gen"] = {
            "accuracy": gen_mc_result["stats"].accuracy,
            "avg_correct_prob": gen_mc_result["stats"].avg_correct_prob,
            "total": gen_mc_result["stats"].total,
        }

    result["mc_eval"] = mc_eval

    with open(combo_dir / "mc_eval.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Baseline / raw-steered evaluation (once per layer)
# ---------------------------------------------------------------------------

def _eval_baselines(
    loaded: LoadedModel,
    val_items: List[dict],
    layer: int,
    base_vector: torch.Tensor,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate baseline (no steering) and raw CAA steering for a layer."""
    baselines: Dict[str, Any] = {}

    # Baseline (no steering)
    bl = evaluate_multiple_choice(
        loaded.model,
        loaded.tokenizer,
        val_items,
        layer_index=layer,
        steering_vector=None,
        scale=0.0,
        primary_device=device,
        seed=seed,
    )
    baselines["baseline"] = {
        "accuracy": bl["stats"].accuracy,
        "avg_correct_prob": bl["stats"].avg_correct_prob,
        "total": bl["stats"].total,
    }

    # Raw CAA steered
    st = evaluate_multiple_choice(
        loaded.model,
        loaded.tokenizer,
        val_items,
        layer_index=layer,
        steering_vector=base_vector.to(device),
        scale=1.0,
        primary_device=device,
        seed=seed,
    )
    baselines["steered"] = {
        "accuracy": st["stats"].accuracy,
        "avg_correct_prob": st["stats"].avg_correct_prob,
        "total": st["stats"].total,
    }

    return baselines


# ---------------------------------------------------------------------------
# Phase 1 orchestrator
# ---------------------------------------------------------------------------

def run_phase1(
    sweep_config: SweepConfig,
    base_config: dict,
    sweep_dir: Path,
    target_layers: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Execute Phase 1: extract + train + MC-eval across the full grid.

    Args:
        target_layers: Subset of layers (for multi-machine parallelism).
                       ``None`` means all layers from *sweep_config*.
    """
    layers = target_layers or sweep_config.layers
    ckpt = SweepCheckpoint(sweep_dir)

    setup_environment(base_config)
    seed = base_config.get("run", {}).get("seed", 42)
    set_random_seeds(seed)

    # Dataset (shared)
    tqa_cfg = base_config.get("truthfulqa", {})
    dataset = TruthfulQADatasetManager(
        dataset_name=tqa_cfg.get("dataset_name", "truthful_qa"),
        dataset_config=tqa_cfg.get("dataset_config", "generation"),
        cache_dir=tqa_cfg.get("cache_dir"),
        seed=seed,
    )

    # Splits (shared, create once)
    splits_path = sweep_dir / "metadata" / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
    else:
        split_cfg = tqa_cfg.get("split", {})
        splits_obj = dataset.create_pipeline_splits(
            steering_pool_size=split_cfg.get("steering_pool", 100),
            train_size=split_cfg.get("train", 250),
            val_size=split_cfg.get("val", 117),
            test_size=split_cfg.get("test", 200),
        )
        splits = {
            "steering_pool": splits_obj.steering_pool,
            "train": splits_obj.train,
            "val": splits_obj.val,
            "test": splits_obj.test,
        }
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=2)

    # Test items for MC evaluation
    # Phase 1 screens on val to avoid optimizer's curse;
    # Phase 2 evaluates on held-out test for unbiased estimates.
    mc_val_indices = [i for i in splits["val"] if dataset.is_valid_mc(i)]
    val_items = dataset.get_items(mc_val_indices)
    LOG.info("Phase 1 MC screening set (val): %d items", len(val_items))

    # Load model ONCE
    model_cfg = base_config["model"]
    LOG.info("Loading model: %s", model_cfg["name"])
    loaded = load_causal_model(
        model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device_map=model_cfg.get("device_map", "auto"),
        max_memory=model_cfg.get("max_memory"),
        revision=model_cfg.get("revision"),
    )
    loaded.model.eval()
    device = loaded.primary_device
    param_dtype = next(loaded.model.parameters()).dtype

    all_results: List[Dict[str, Any]] = []

    for layer in layers:
        LOG.info("=" * 60)
        LOG.info("LAYER %d", layer)
        LOG.info("=" * 60)

        # -- Extract --
        if ckpt.is_layer_extracted(layer):
            LOG.info("Layer %d already extracted, loading from cache", layer)
            vector_bank = _load_vector_bank(sweep_dir, layer)
        else:
            vector_bank = run_layer_extraction(
                loaded, dataset, splits["steering_pool"],
                layer, sweep_dir, base_config,
            )
            ckpt.mark_layer_extracted(layer)

        # -- Baselines (once per layer) --
        base_vec = vector_bank.base_vector.to(device, dtype=param_dtype)
        baselines = _eval_baselines(
            loaded, val_items, layer, base_vec, device, seed,
        )
        bl_dir = sweep_dir / "results" / f"layer_{layer:02d}"
        bl_dir.mkdir(parents=True, exist_ok=True)
        with open(bl_dir / "baselines.json", "w") as f:
            json.dump(baselines, f, indent=2)

        LOG.info(
            "  Baselines — no-steer: %.1f%%, raw CAA: %.1f%%",
            baselines["baseline"]["accuracy"] * 100,
            baselines["steered"]["accuracy"] * 100,
        )

        # -- HP combos --
        combos = sweep_config.configs_for_layer(layer)
        for i, combo in enumerate(combos):
            cid = combo["combo_id"]

            if ckpt.is_combo_complete(layer, cid):
                LOG.info(
                    "  [%d/%d] %s already complete, skipping",
                    i + 1, len(combos), cid,
                )
                cached = ckpt.get_all_results().get(f"{layer}/{cid}")
                if cached:
                    all_results.append(cached)
                continue

            LOG.info(
                "  [%d/%d] %s  (lr=%s, reg=%s)",
                i + 1, len(combos), cid, combo["lr"], combo["mse_reg"],
            )

            res = run_hp_combo(
                loaded, dataset, splits["train"], val_items,
                vector_bank, layer, combo["lr"], combo["mse_reg"],
                sweep_dir, base_config,
            )

            # Attach baseline numbers for easy comparison
            res["baselines"] = baselines
            ckpt.mark_combo_complete(layer, cid, res)
            all_results.append(res)

            mc_mc = res.get("mc_eval", {}).get("mlp_mc", {}).get("accuracy", 0)
            mc_gen = res.get("mc_eval", {}).get("mlp_gen", {}).get("accuracy", 0)
            train_acc = res.get("mc_final_acc", 0) or 0
            LOG.info(
                "    train: %.0f%% → val_mc: %.1f%%  val_gen: %.1f%%  (gap: %.0f%%)",
                train_acc * 100, mc_mc * 100, mc_gen * 100,
                (train_acc - mc_mc) * 100,
            )

    # Save aggregated Phase 1 results
    with open(sweep_dir / "phase1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
