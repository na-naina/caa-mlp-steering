#!/usr/bin/env python3
"""Stage 2: Train MLP steering networks.

Resource requirements: Forward + backward pass (needs gradients)
- Higher VRAM than extraction due to gradient storage
- ~44GB for 12B models with batch_size=4
- Reduce batch sizes if OOM

Inputs (from extract stage):
- vectors/vector_bank.pt

Outputs:
- vectors/mlp_mc_state_dict.pt
- vectors/mlp_gen_state_dict.pt
- metadata/train.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from src.data.truthfulqa import TruthfulQADatasetManager
from src.models.loader import load_causal_model
from src.stages.common import (
    CheckpointManager,
    RunContext,
    check_stage_complete,
    get_or_create_run,
    load_config,
    load_stage_metadata,
    save_stage_metadata,
    set_random_seeds,
    setup_environment,
    setup_logging,
)
from src.steering.mlp import SteeringMLP
from src.steering.training import (
    GenTrainingConfig,
    MCTrainingConfig,
    train_gen_mlp,
    train_mc_mlp,
)
from src.steering.vector_bank import VectorBank

LOG = logging.getLogger(__name__)


def load_vector_bank(ctx: RunContext) -> VectorBank:
    """Load vector bank from extraction stage."""
    vb_path = ctx.vectors_dir / "vector_bank.pt"
    if not vb_path.exists():
        raise FileNotFoundError(
            f"Vector bank not found at {vb_path}. Run extraction stage first."
        )

    data = torch.load(vb_path)
    return VectorBank(
        base_vector=data["base_vector"],
        vectors=data["vectors"],
        indices=data["indices"],
    )


def load_splits(ctx: RunContext) -> dict:
    """Load data splits from extraction stage."""
    splits_path = ctx.metadata_dir / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits not found at {splits_path}. Run extraction stage first."
        )

    with open(splits_path) as f:
        return json.load(f)


def run_training(ctx: RunContext, force: bool = False) -> dict:
    """Train MC and generation MLPs with checkpointing.

    Training is checkpointed after each MLP completes, so if the job
    is interrupted, it can resume from the last completed MLP.
    """
    config = ctx.config
    model_cfg = config["model"]
    mlp_cfg = config.get("mlp", {})
    steering_cfg = config.get("steering", {})
    seed = config.get("run", {}).get("seed", 42)

    set_random_seeds(seed)

    # Initialize checkpoint manager
    ckpt = CheckpointManager(ctx, "train")
    if force:
        ckpt.clear()

    # Load prerequisites
    vector_bank = load_vector_bank(ctx)
    splits = load_splits(ctx)

    # Check vector bank validity
    base_norm = vector_bank.base_vector.norm().item()
    if base_norm < 1e-6:
        LOG.error(
            "Base vector has near-zero norm (%.2e). "
            "Training will likely fail. Check extraction stage.",
            base_norm
        )

    # Load model (needs gradients for MLP backward pass through steering hook)
    LOG.info("Loading model: %s", model_cfg["name"])
    loaded = load_causal_model(
        model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device_map=model_cfg.get("device_map", "auto"),
        max_memory=model_cfg.get("max_memory"),
        revision=model_cfg.get("revision"),
    )
    model = loaded.model
    model.eval()  # Model stays in eval, only MLP trains

    device = loaded.primary_device
    param_dtype = next(model.parameters()).dtype

    # Load dataset
    tqa_cfg = config.get("truthfulqa", {})
    dataset = TruthfulQADatasetManager(
        dataset_name=tqa_cfg.get("dataset_name", "truthful_qa"),
        dataset_config=tqa_cfg.get("dataset_config", "generation"),
        cache_dir=tqa_cfg.get("cache_dir"),
        seed=seed,
    )

    hidden_dim = vector_bank.base_vector.shape[0]
    arch_cfg = mlp_cfg.get("architecture", {})
    mc_history = {}
    gen_history = {}
    mc_valid = False
    gen_valid = False

    # ===== Train MC MLP (with checkpointing) =====
    if ckpt.is_complete("mlp_mc"):
        LOG.info("MC MLP already trained (resuming from checkpoint)")
        mc_result = ckpt.get_result("mlp_mc")
        mc_valid = mc_result.get("valid", False)
        mc_history = mc_result.get("history", {})
    else:
        LOG.info("Training MC MLP")
        mlp_mc = SteeringMLP(
            input_dim=hidden_dim,
            hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
            dropout=arch_cfg.get("dropout", 0.1),
        ).to(device, dtype=param_dtype)

        mc_cfg = MCTrainingConfig(**mlp_cfg.get("mc_training", {}))
        mc_history = train_mc_mlp(
            mlp_mc,
            model=model,
            tokenizer=loaded.tokenizer,
            dataset=dataset,
            train_indices=splits["train"],
            vector_bank=vector_bank,
            layer_index=model_cfg["layer"],
            primary_device=device,
            max_length=steering_cfg.get("max_length", 512),
            config=mc_cfg,
            seed=seed + 1,
        )

        # Check for NaN in MC training
        mc_valid = bool(mc_history.get("loss")) and not any(
            v != v for v in mc_history.get("loss", [])  # NaN check
        )

        if mc_valid:
            torch.save(mlp_mc.state_dict(), ctx.vectors_dir / "mlp_mc_state_dict.pt")
            LOG.info("MC MLP saved (final loss: %.4f)", mc_history["loss"][-1])
        else:
            LOG.warning("MC MLP training produced NaN - not saving")

        # Checkpoint MC MLP completion
        ckpt.mark_complete("mlp_mc", {
            "valid": mc_valid,
            "history": mc_history,
            "final_loss": mc_history["loss"][-1] if mc_history.get("loss") else None,
            "final_acc": mc_history["accuracy"][-1] if mc_history.get("accuracy") else None,
        })

    # ===== Train Gen MLP (with checkpointing) =====
    if ckpt.is_complete("mlp_gen"):
        LOG.info("Gen MLP already trained (resuming from checkpoint)")
        gen_result = ckpt.get_result("mlp_gen")
        gen_valid = gen_result.get("valid", False)
        gen_history = gen_result.get("history", {})
    else:
        LOG.info("Training Gen MLP")
        mlp_gen = SteeringMLP(
            input_dim=hidden_dim,
            hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
            dropout=arch_cfg.get("dropout", 0.1),
        ).to(device, dtype=param_dtype)

        gen_cfg = GenTrainingConfig(**mlp_cfg.get("gen_training", {}))
        gen_history = train_gen_mlp(
            mlp_gen,
            model=model,
            tokenizer=loaded.tokenizer,
            dataset=dataset,
            train_indices=splits["train"],
            vector_bank=vector_bank,
            layer_index=model_cfg["layer"],
            primary_device=device,
            max_length=steering_cfg.get("max_length", 512),
            config=gen_cfg,
            seed=seed + 2,
        )

        # Check for NaN in Gen training
        gen_valid = bool(gen_history.get("loss")) and not any(
            v != v for v in gen_history.get("loss", [])
        )

        if gen_valid:
            torch.save(mlp_gen.state_dict(), ctx.vectors_dir / "mlp_gen_state_dict.pt")
            LOG.info("Gen MLP saved (final loss: %.4f)", gen_history["loss"][-1])
        else:
            LOG.warning("Gen MLP training produced NaN - not saving")

        # Checkpoint Gen MLP completion
        ckpt.mark_complete("mlp_gen", {
            "valid": gen_valid,
            "history": gen_history,
            "final_loss": gen_history["loss"][-1] if gen_history.get("loss") else None,
        })

    # Save training history
    with open(ctx.metadata_dir / "training_history.json", "w") as f:
        json.dump({"mc": mc_history, "gen": gen_history}, f, indent=2)

    metadata = {
        "model": model_cfg["name"],
        "mc_valid": mc_valid,
        "gen_valid": gen_valid,
        "mc_final_loss": mc_history.get("loss", [None])[-1] if mc_history.get("loss") else None,
        "gen_final_loss": gen_history.get("loss", [None])[-1] if gen_history.get("loss") else None,
        "mc_final_acc": mc_history.get("accuracy", [None])[-1] if mc_history.get("accuracy") else None,
        "hidden_dim": hidden_dim,
    }

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train MLP steering networks")
    parser.add_argument("--model", required=True, help="Model config name")
    parser.add_argument("--run-id", required=True, help="Run ID from extraction stage")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if complete")
    args = parser.parse_args()

    config = load_config(args.model, args.config_dir)
    setup_environment(config)

    ctx = get_or_create_run(args.model, config, args.run_id)
    setup_logging(args.verbose, ctx.run_dir / "logs" / "train.log")

    LOG.info("Run directory: %s", ctx.run_dir)

    # Check prerequisites
    if not check_stage_complete(ctx, "extract"):
        LOG.error("Extraction stage not complete. Run extract stage first.")
        return 1

    if check_stage_complete(ctx, "train") and not args.force:
        LOG.info("Training already complete, skipping (use --force to re-run)")
        return 0

    try:
        metadata = run_training(ctx, force=args.force)
        save_stage_metadata(ctx, "train", metadata)
        LOG.info("Training stage complete")
        return 0
    except Exception as e:
        LOG.exception("Training failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
