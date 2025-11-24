from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.data.truthfulqa import (
    TruthfulQADatasetManager,
    TruthfulQAPipelineSplits,
)
from src.evaluation.judge import LLMBinaryJudge
from src.evaluation.semantic import SemanticJudge, SemanticJudgeConfig
from src.evaluation.informativeness import LLMInformativenessJudge
from src.evaluation.finetuned_judge import FinetunedJudge
from src.evaluation.truthfulqa import (
    evaluate_generation,
    evaluate_multiple_choice,
)
from src.models.loader import load_causal_model, _parse_dtype
from src.steering.extract import (
    ActivationExtractor,
    compute_caa_vector,
)
from src.steering.mlp import SteeringMLP
from src.steering.training import (
    MCTrainingConfig,
    GenTrainingConfig,
    train_gen_mlp,
    train_mc_mlp,
)
from src.steering.vector_bank import VectorBankBuilder
from src.utils.config import dump_config, load_config

LOGGER = logging.getLogger("caa.pipeline")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=level,
    )


def _serialize_for_json(obj: Any) -> Any:
    """Recursively convert dataclass objects to dicts for JSON serialization."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma CAA pipeline")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Base YAML configuration",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Model-specific YAML configuration",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run identifier",
    )
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Additional config overrides (key=value)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip heavy steps (model loading) to verify configuration",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def ensure_hf_token() -> None:
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token_path.exists():
        LOGGER.warning("HuggingFace token file not found at %s", token_path)
        return

    token = token_path.read_text().strip()
    if not token:
        LOGGER.warning("HuggingFace token file is empty")
        return

    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    LOGGER.info("Configured HuggingFace authentication from %s", token_path)


def set_random_seeds(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_cache_paths(paths_cfg: Dict) -> None:
    hf_cache = paths_cfg.get("hf_cache")
    if hf_cache:
        cache_path = Path(hf_cache)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_path))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path / "transformers"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path / "datasets"))
    cache_root = paths_cfg.get("cache_root")
    if cache_root:
        cache_path = Path(cache_root)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path / "datasets"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path / "transformers"))


def prepare_output_dir(config: Dict) -> Path:
    output_root = Path(config["paths"]["output_root"])
    run_id = config["run"]["id"]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    ensure_hf_token()

    config = load_config(
        args.base_config,
        overrides=[args.config],
        cli_overrides=args.override,
    )
    if args.run_id:
        config.setdefault("run", {})["id"] = args.run_id

    _configure_cache_paths(config.get("paths", {}))

    LOGGER.info(
        "Cache environment: HF_HOME=%s HF_DATASETS_CACHE=%s TRANSFORMERS_CACHE=%s",
        os.environ.get("HF_HOME"),
        os.environ.get("HF_DATASETS_CACHE"),
        os.environ.get("TRANSFORMERS_CACHE"),
    )

    set_random_seeds(config["run"].get("seed", 42))
    run_dir = prepare_output_dir(config)
    dump_config(config, run_dir / "config.yaml")
    LOGGER.info("Run directory: %s", run_dir)

    if args.dry_run:
        LOGGER.info("Dry run complete (configuration only)")
        return 0

    dataset_manager = TruthfulQADatasetManager(
        dataset_name=config["truthfulqa"].get("dataset_name", "truthful_qa"),
        dataset_config=config["truthfulqa"].get("dataset_config", "generation"),
        cache_dir=config["truthfulqa"].get("cache_dir"),
        seed=config["run"].get("seed", 42),
    )

    splits = _prepare_pipeline_splits(dataset_manager, config)
    _persist_splits(run_dir, splits)

    model_cfg = config["model"]
    loaded = load_causal_model(
        model_cfg["name"],
        dtype=model_cfg.get("dtype"),
        device_map=model_cfg.get("device_map", "auto"),
        max_memory=model_cfg.get("max_memory"),
        revision=model_cfg.get("revision"),
    )
    model = loaded.model
    tokenizer = loaded.tokenizer
    primary_device = loaded.primary_device
    model.eval()

    # Log GPU allocation for debugging
    dm = getattr(model, "hf_device_map", None)
    if dm:
        used_devs = set(dm.values())
        LOGGER.info("GPU allocation: main model uses %s", sorted(used_devs))

    steering_cfg = config.get("steering", {})
    safe_attention = steering_cfg.get("safe_attention", False)
    autocast_cfg = steering_cfg.get("autocast_dtype")
    autocast_dtype = _parse_dtype(autocast_cfg) if autocast_cfg else None
    extractor = ActivationExtractor(
        loaded,
        model_cfg["layer"],
        max_length=steering_cfg.get("max_length", 512),
        batch_size=steering_cfg.get("batch_size", 8),
        safe_attention=safe_attention,
        autocast_dtype=autocast_dtype,
    )

    vector_bank = _build_vector_bank(
        extractor,
        dataset_manager,
        splits,
        steering_cfg,
        seed=config["run"].get("seed", 42),
        run_dir=run_dir,
    )

    mlp_results = {}

    use_mlp = steering_cfg.get("use_mlp", True)
    mlp_mc = None
    mlp_gen = None

    if use_mlp:
        hidden_dim = vector_bank.base_vector.shape[0]
        arch_cfg = config.get("mlp", {}).get("architecture", {})
        param_dtype = next(model.parameters()).dtype  # Get model's dtype
        mlp_mc = SteeringMLP(
            input_dim=hidden_dim,
            hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
            dropout=arch_cfg.get("dropout", 0.1),
        ).to(primary_device, dtype=param_dtype)  # Match model dtype
        mlp_gen = SteeringMLP(
            input_dim=hidden_dim,
            hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
            dropout=arch_cfg.get("dropout", 0.1),
        ).to(primary_device, dtype=param_dtype)  # Match model dtype

        mlp_cfg = config.get("mlp", {})
        mc_cfg = MCTrainingConfig(**mlp_cfg.get("mc_training", {}))
        gen_cfg = GenTrainingConfig(**mlp_cfg.get("gen_training", {}))

        LOGGER.info("Training MC MLP: %s", mc_cfg)
        mc_history = train_mc_mlp(
            mlp_mc,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset_manager,
            train_indices=splits.train,
            vector_bank=vector_bank,
            layer_index=model_cfg["layer"],
            primary_device=primary_device,
            max_length=steering_cfg.get("max_length", 512),
            config=mc_cfg,
            seed=config["run"].get("seed", 42) + 1,
        )
        mlp_results["mc"] = mc_history
        if mc_history.get("loss"):
            torch.save(
                mlp_mc.state_dict(),
                run_dir / "vectors" / "mlp_mc_state_dict.pt",
            )
        else:
            mlp_mc = None

        LOGGER.info("Training Generation MLP: %s", gen_cfg)
        gen_history = train_gen_mlp(
            mlp_gen,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset_manager,
            train_indices=splits.train,
            vector_bank=vector_bank,
            layer_index=model_cfg["layer"],
            primary_device=primary_device,
            max_length=steering_cfg.get("max_length", 512),
            config=gen_cfg,
            seed=config["run"].get("seed", 42) + 2,
        )
        mlp_results["gen"] = gen_history
        torch.save(
            mlp_gen.state_dict(), run_dir / "vectors" / "mlp_gen_state_dict.pt"
        )

        with (run_dir / "training_history.json").open("w") as f:
            json.dump(mlp_results, f, indent=2)

    # First generate all responses, then free model, then load judge
    # This allows us to use 3 GPUs for model, then reuse them for judge
    LOGGER.info("Starting evaluation: generating all responses first")
    evaluation = _run_evaluations(
        model,
        tokenizer,
        dataset_manager,
        splits,
        vector_bank,
        mlp_mc,
        mlp_gen,
        config,
        judge=None,  # Don't load judge yet
        semantic_judge=None,
        primary_device=primary_device,
        run_dir=run_dir,
    )

    # Free the main model to make space for judge
    LOGGER.info("Freeing main model to load judge")
    del model
    del loaded
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Now load all judges and score all generated responses
    eval_cfg = config.get("evaluation", {})

    # Check if we can share the judge model to save memory
    shared_judge_model = None
    judge_cfg = eval_cfg.get("judge", {})
    info_cfg = eval_cfg.get("informativeness", {})

    truth_model = judge_cfg.get("model") if judge_cfg.get("mode", "zero_shot") == "zero_shot" else None
    info_model = info_cfg.get("model") if info_cfg.get("mode", "zero_shot") == "zero_shot" and info_cfg.get("enabled", False) else None

    # If both judges use the same model, load it once
    if truth_model and info_model and truth_model == info_model:
        LOGGER.info("Both judges use '%s' - loading once and sharing", truth_model)
        shared_judge_model = load_causal_model(
            truth_model,
            dtype=judge_cfg.get("dtype", "bfloat16"),
            device_map=judge_cfg.get("device_map", "auto"),
        )

    judge = _maybe_build_judge(eval_cfg, shared_model=shared_judge_model)
    informativeness_judge = _maybe_build_informativeness(eval_cfg, shared_model=shared_judge_model)
    semantic_judge = _maybe_build_semantic(eval_cfg)
    bleurt_judge = _maybe_build_bleurt(eval_cfg)

    if judge or informativeness_judge or semantic_judge or bleurt_judge:
        LOGGER.info("Scoring generated responses with judges")
        evaluation = _score_all_evaluations(
            evaluation, judge, informativeness_judge, semantic_judge, bleurt_judge, run_dir
        )

    with (run_dir / "results.json").open("w") as f:
        json.dump(_serialize_for_json(evaluation), f, indent=2)

    LOGGER.info("Experiment complete - results stored in %s", run_dir)
    return 0


def _prepare_pipeline_splits(
    dataset_manager: TruthfulQADatasetManager,
    config: Dict,
) -> TruthfulQAPipelineSplits:
    split_cfg = config["truthfulqa"].get("split", {})
    splits = dataset_manager.create_pipeline_splits(
        steering_pool_size=split_cfg.get("steering_pool", 100),
        train_size=split_cfg.get("train", 250),
        val_size=split_cfg.get("val", 117),
        test_size=split_cfg.get("test", 200),
    )
    return splits


def _persist_splits(run_dir: Path, splits: TruthfulQAPipelineSplits) -> None:
    (run_dir / "metadata").mkdir(exist_ok=True)
    with (run_dir / "metadata" / "splits.json").open("w") as f:
        json.dump(
            {
                "steering_pool": splits.steering_pool,
                "train": splits.train,
                "val": splits.val,
                "test": splits.test,
            },
            f,
            indent=2,
        )


def _build_vector_bank(
    extractor: ActivationExtractor,
    dataset_manager: TruthfulQADatasetManager,
    splits: TruthfulQAPipelineSplits,
    steering_cfg: Dict,
    *,
    seed: int,
    run_dir: Path,
):
    pool_prompts_pos, pool_prompts_neg, valid_indices = dataset_manager.build_caa_prompts(
        splits.steering_pool
    )

    if len(valid_indices) < len(pool_prompts_pos):
        LOGGER.warning(
            "Steering pool reduced to %d valid items (requested %d)",
            len(valid_indices),
            len(splits.steering_pool),
        )

    LOGGER.info("Collecting steering activations (%d prompts)", len(pool_prompts_pos))
    pos_acts, pos_valid_indices = extractor.collect_mean_activations(pool_prompts_pos)
    neg_acts, neg_valid_indices = extractor.collect_mean_activations(pool_prompts_neg)

    # Keep only pairs where both positive and negative are valid
    valid_pair_indices = sorted(set(pos_valid_indices) & set(neg_valid_indices))

    if len(valid_pair_indices) < len(pool_prompts_pos):
        num_skipped = len(pool_prompts_pos) - len(valid_pair_indices)
        LOGGER.warning(
            f"Skipped {num_skipped} pairs due to NaN/Inf activations, "
            f"using {len(valid_pair_indices)} valid pairs"
        )

    # Filter to keep only valid pairs
    pos_mask = torch.tensor([i in valid_pair_indices for i in pos_valid_indices])
    neg_mask = torch.tensor([i in valid_pair_indices for i in neg_valid_indices])
    pos_acts = pos_acts[pos_mask]
    neg_acts = neg_acts[neg_mask]

    if len(pos_acts) == 0:
        raise RuntimeError("No valid activation pairs remaining after NaN/Inf filtering")

    vector_dir = run_dir / "vectors"
    vector_dir.mkdir(exist_ok=True)

    base_vector = compute_caa_vector(pos_acts, neg_acts, normalize=True)
    torch.save(base_vector.detach().cpu(), vector_dir / "base_vector.pt")

    bank_cfg = steering_cfg.get("vector_bank", {})
    builder = VectorBankBuilder(
        pos_acts,
        neg_acts,
        normalize=True,
        seed=seed,
    )
    bank = builder.build(
        num_vectors=bank_cfg.get("num_vectors", 16),
        sample_size_range=(
            bank_cfg.get("min_samples", 30),
            bank_cfg.get("max_samples", 50),
        ),
    )

    torch.save(
        {
            "base_vector": bank.base_vector.detach().cpu(),
            "vectors": [v.detach().cpu() for v in bank.vectors],
            "indices": bank.indices,
        },
        vector_dir / "vector_bank.pt",
    )
    return bank


def _maybe_build_judge(eval_cfg: Dict, shared_model=None):
    """Build truthfulness judge (zero-shot or fine-tuned)."""
    judge_cfg = eval_cfg.get("judge", {})
    mode = judge_cfg.get("mode", "zero_shot")

    if mode == "zero_shot":
        model_name = judge_cfg.get("model")
        if not model_name:
            LOGGER.info("No LLM judge configured; skipping")
            return None
        return LLMBinaryJudge(
            model_name,
            dtype=judge_cfg.get("dtype", "bfloat16"),
            device_map=judge_cfg.get("device_map", "auto"),
            max_new_tokens=judge_cfg.get("max_new_tokens", 32),
            shared_model=shared_model,
        )
    elif mode == "finetuned":
        model_name = judge_cfg.get("finetuned_model")
        if not model_name:
            LOGGER.warning("Fine-tuned judge mode selected but no model specified; skipping")
            return None
        return FinetunedJudge(
            model_name,
            mode="truth",
            dtype=judge_cfg.get("dtype", "bfloat16"),
            device_map=judge_cfg.get("device_map", "auto"),
            threshold=judge_cfg.get("threshold", 0.5),
        )
    else:
        LOGGER.warning("Unknown judge mode '%s'; skipping", mode)
        return None


def _maybe_build_informativeness(eval_cfg: Dict, shared_model=None):
    """Build informativeness judge (zero-shot or fine-tuned)."""
    info_cfg = eval_cfg.get("informativeness", {})
    if not info_cfg.get("enabled", False):
        LOGGER.info("Informativeness judge disabled; skipping")
        return None

    mode = info_cfg.get("mode", "zero_shot")

    if mode == "zero_shot":
        model_name = info_cfg.get("model")
        if not model_name:
            LOGGER.warning("Informativeness judge enabled but no model specified; skipping")
            return None
        return LLMInformativenessJudge(
            model_name,
            dtype=info_cfg.get("dtype", "bfloat16"),
            device_map=info_cfg.get("device_map", "auto"),
            max_new_tokens=info_cfg.get("max_new_tokens", 32),
            shared_model=shared_model,
        )
    elif mode == "finetuned":
        model_name = info_cfg.get("finetuned_model")
        if not model_name:
            LOGGER.warning("Fine-tuned informativeness mode selected but no model specified; skipping")
            return None
        return FinetunedJudge(
            model_name,
            mode="info",
            dtype=info_cfg.get("dtype", "bfloat16"),
            device_map=info_cfg.get("device_map", "auto"),
            threshold=info_cfg.get("threshold", 0.5),
        )
    else:
        LOGGER.warning("Unknown informativeness mode '%s'; skipping", mode)
        return None


def _maybe_build_semantic(eval_cfg: Dict):
    semantic_cfg = eval_cfg.get("semantic", {})
    if not semantic_cfg.get("enabled", True):
        return None
    config = SemanticJudgeConfig(
        model_name=semantic_cfg.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        similarity_threshold=semantic_cfg.get("similarity_threshold", 0.6),
    )
    return SemanticJudge(config)


def _maybe_build_bleurt(eval_cfg: Dict):
    """Build BLEURT judge if enabled."""
    from src.evaluation.bleurt_judge import BLEURTJudge, BLEURTJudgeConfig

    bleurt_cfg = eval_cfg.get("bleurt", {})
    if not bleurt_cfg.get("enabled", False):
        LOGGER.info("BLEURT judge disabled; skipping")
        return None

    try:
        config = BLEURTJudgeConfig(
            checkpoint=bleurt_cfg.get("checkpoint", "bleurt-base-128"),
            cache_dir=bleurt_cfg.get("cache_dir"),
        )
        return BLEURTJudge(config)
    except Exception as exc:
        LOGGER.error("Failed to initialize BLEURT judge: %s", exc)
        return None


def _score_all_evaluations(
    evaluation: Dict,
    judge,
    informativeness_judge,
    semantic_judge,
    bleurt_judge,
    run_dir: Path,
) -> Dict:
    """Score all generated responses with judges after model is freed."""
    from src.evaluation.truthfulqa import _summarize_generation

    for variant_name, variant_data in evaluation.items():
        for scale_key, scale_data in variant_data.items():
            gen_data = scale_data.get("generation")
            if gen_data and "details" in gen_data:
                details = gen_data["details"]

                # Score with all judges
                annotated = details
                if semantic_judge:
                    LOGGER.info("Applying semantic judge to %s/%s", variant_name, scale_key)
                    annotated = semantic_judge.score_responses(annotated)
                if judge:
                    LOGGER.info("Applying truthfulness judge to %s/%s", variant_name, scale_key)
                    annotated = judge.score_responses(annotated)
                if informativeness_judge:
                    LOGGER.info("Applying informativeness judge to %s/%s", variant_name, scale_key)
                    annotated = informativeness_judge.score_responses(annotated)
                if bleurt_judge:
                    LOGGER.info("Applying BLEURT judge to %s/%s", variant_name, scale_key)
                    annotated = bleurt_judge.score_responses(annotated)

                # Recompute stats with judge scores
                stats = _summarize_generation(
                    annotated,
                    judge is not None,
                    informativeness_judge is not None,
                    semantic_judge is not None,
                    bleurt_judge is not None,
                )

                # Update the in-memory data - flatten the stats into top level
                gen_data["details"] = annotated
                # Convert dataclass to dict and merge into gen_data
                stats_dict = dataclasses.asdict(stats)
                gen_data.update(stats_dict)

                # Also update the individual generation_details.json file on disk
                detail_file = run_dir / variant_name / scale_key / "generation_details.json"
                if detail_file.exists():
                    with detail_file.open("w") as f:
                        json.dump(annotated, f, indent=2)

    return evaluation


def _run_evaluations(
    model,
    tokenizer,
    dataset_manager: TruthfulQADatasetManager,
    splits: TruthfulQAPipelineSplits,
    vector_bank,
    mlp_mc: Optional[SteeringMLP],
    mlp_gen: Optional[SteeringMLP],
    config: Dict,
    judge,
    semantic_judge,
    primary_device,
    run_dir: Path,
):
    steering_cfg = config.get("steering", {})
    scales = steering_cfg.get("scales", [1.0])

    eval_cfg = config.get("evaluation", {})
    gen_cfg = {
        "preset": eval_cfg.get("preset"),  # TruthfulQA preset (qa, help, null, etc.)
        "temperature": eval_cfg.get("temperature", 0.7),
        "top_p": eval_cfg.get("top_p", 0.9),
        "top_k": eval_cfg.get("top_k", 50),
        "max_new_tokens": eval_cfg.get("max_new_tokens", 80),
        "max_length": steering_cfg.get("max_length", 512),
        "stop_sequences": eval_cfg.get("stop_sequences", []),
    }

    test_items = dataset_manager.get_items(splits.test)
    test_mc_indices = [idx for idx in splits.test if dataset_manager.is_valid_mc(idx)]
    mc_items = dataset_manager.get_items(test_mc_indices)

    param_dtype = next(model.parameters()).dtype

    # Build all available variants
    all_variants = {
        "baseline": None,
        "steered": vector_bank.base_vector.to(primary_device, dtype=param_dtype),
    }

    if mlp_mc is not None:
        base = vector_bank.base_vector.unsqueeze(0).to(primary_device, dtype=param_dtype)
        vector = mlp_mc(base).squeeze(0)
        all_variants["mlp_mc"] = vector.detach()
    if mlp_gen is not None:
        base = vector_bank.base_vector.unsqueeze(0).to(primary_device, dtype=param_dtype)
        vector = mlp_gen(base).squeeze(0)
        all_variants["mlp_gen"] = vector.detach()

    # Filter variants based on config
    enabled_variants = steering_cfg.get("enabled_variants", None)
    if enabled_variants is not None:
        variants = {k: v for k, v in all_variants.items() if k in enabled_variants}
        LOGGER.info("Using enabled variants: %s", list(variants.keys()))
    else:
        variants = all_variants
        LOGGER.info("Using all available variants: %s", list(variants.keys()))

    layer_index = config["model"]["layer"]

    evaluation: Dict[str, Dict[str, Dict]] = {}

    for variant, vector in variants.items():
        variant_results: Dict[str, Dict] = {}
        variant_dir = run_dir / variant
        variant_dir.mkdir(exist_ok=True)

        for scale in scales if vector is not None else [0.0]:
            mc_result = evaluate_multiple_choice(
                model,
                tokenizer,
                mc_items,
                layer_index=layer_index,
                steering_vector=vector,
                scale=scale,
                max_length=steering_cfg.get("max_length", 512),
                primary_device=primary_device,
                seed=config["run"].get("seed", 42),
            )
            gen_result = evaluate_generation(
                model,
                tokenizer,
                test_items,
                layer_index=layer_index,
                steering_vector=vector,
                scale=scale,
                generation_cfg=gen_cfg,
                primary_device=primary_device,
                judge=judge,
                semantic_judge=semantic_judge,
            )

            key = f"scale_{scale:.2f}"
            variant_results[key] = {
                "mc": _serialize_mc_stats(mc_result),
                "generation": _serialize_generation_stats(gen_result),
            }

            scale_dir = variant_dir / key
            scale_dir.mkdir(exist_ok=True)
            with (scale_dir / "mc_details.json").open("w") as f:
                json.dump(mc_result["details"], f, indent=2)
            with (scale_dir / "generation_details.json").open("w") as f:
                json.dump(gen_result["details"], f, indent=2)

        evaluation[variant] = variant_results

    return evaluation


def _serialize_mc_stats(mc_result: Dict) -> Dict:
    stats = mc_result["stats"]
    return {
        "accuracy": stats.accuracy,
        "avg_correct_prob": stats.avg_correct_prob,
        "avg_incorrect_prob": stats.avg_incorrect_prob,
        "total": stats.total,
    }


def _serialize_generation_stats(gen_result: Dict) -> Dict:
    stats = gen_result["stats"]
    # Preserve all GenerationStats fields (e.g., informativeness, semantic diff).
    stats_dict = (
        dataclasses.asdict(stats)
        if dataclasses.is_dataclass(stats)
        else dict(stats)
    )
    stats_dict["details"] = gen_result.get("details", [])
    return stats_dict


if __name__ == "__main__":
    raise SystemExit(main())
