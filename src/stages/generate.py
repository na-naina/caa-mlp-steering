#!/usr/bin/env python3
"""Stage 3: Generate responses with steering vectors.

Resource requirements: Inference only (no gradients)
- Same as extraction stage
- Can skip MLP variants if training failed

Inputs:
- vectors/base_vector.pt
- vectors/mlp_mc_state_dict.pt (optional)
- vectors/mlp_gen_state_dict.pt (optional)
- metadata/splits.json

Outputs:
- responses/{variant}/generation.json
- responses/{variant}/mc.json
- metadata/generate.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.data.truthfulqa import TruthfulQADatasetManager
from src.evaluation.truthfulqa import evaluate_generation, evaluate_multiple_choice
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

LOG = logging.getLogger(__name__)


def load_steering_vectors(ctx: RunContext, device: torch.device, dtype: torch.dtype) -> Dict[str, Optional[torch.Tensor]]:
    """Load all available steering vectors."""
    vectors = {"baseline": None}

    # Load base vector
    base_path = ctx.vectors_dir / "base_vector.pt"
    if base_path.exists():
        base = torch.load(base_path).to(device, dtype=dtype)
        vectors["steered"] = base

        # Load MLP variants if available
        hidden_dim = base.shape[0]

        mc_path = ctx.vectors_dir / "mlp_mc_state_dict.pt"
        if mc_path.exists():
            mlp = SteeringMLP(input_dim=hidden_dim).to(device, dtype=dtype)
            mlp.load_state_dict(torch.load(mc_path))
            mlp.eval()
            with torch.no_grad():
                vectors["mlp_mc"] = mlp(base.unsqueeze(0)).squeeze(0)

        gen_path = ctx.vectors_dir / "mlp_gen_state_dict.pt"
        if gen_path.exists():
            mlp = SteeringMLP(input_dim=hidden_dim).to(device, dtype=dtype)
            mlp.load_state_dict(torch.load(gen_path))
            mlp.eval()
            with torch.no_grad():
                vectors["mlp_gen"] = mlp(base.unsqueeze(0)).squeeze(0)

    return vectors


def run_generation(ctx: RunContext, variants: Optional[List[str]] = None, force: bool = False) -> dict:
    """Generate responses for all steering variants with checkpointing.

    Each variant is checkpointed after completion, so if interrupted,
    the job can resume from the last completed variant.
    """
    config = ctx.config
    model_cfg = config["model"]
    steering_cfg = config.get("steering", {})
    eval_cfg = config.get("evaluation", {})
    seed = config.get("run", {}).get("seed", 42)

    set_random_seeds(seed)

    # Initialize checkpoint manager
    ckpt = CheckpointManager(ctx, "generate")
    if force:
        ckpt.clear()

    # Load splits
    with open(ctx.metadata_dir / "splits.json") as f:
        splits = json.load(f)

    # Load model
    LOG.info("Loading model: %s", model_cfg["name"])
    loaded = load_causal_model(
        model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device_map=model_cfg.get("device_map", "auto"),
        max_memory=model_cfg.get("max_memory"),
        revision=model_cfg.get("revision"),
    )
    model = loaded.model
    model.eval()
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

    # Load steering vectors
    all_vectors = load_steering_vectors(ctx, device, param_dtype)

    # Filter to requested variants
    if variants:
        all_vectors = {k: v for k, v in all_vectors.items() if k in variants}

    # Also filter by config
    enabled = steering_cfg.get("enabled_variants")
    if enabled:
        all_vectors = {k: v for k, v in all_vectors.items() if k in enabled}

    # Get test items
    test_items = dataset.get_items(splits["test"])
    mc_indices = [i for i in splits["test"] if dataset.is_valid_mc(i)]
    mc_items = dataset.get_items(mc_indices)

    # Generation config
    gen_cfg = {
        "preset": eval_cfg.get("preset", "qa"),
        "temperature": eval_cfg.get("temperature", 0.3),
        "top_p": eval_cfg.get("top_p", 0.9),
        "top_k": eval_cfg.get("top_k", 50),
        "max_new_tokens": eval_cfg.get("max_new_tokens", 64),
        "max_length": steering_cfg.get("max_length", 512),
        "stop_sequences": eval_cfg.get("stop_sequences", []),
    }

    layer_index = model_cfg["layer"]
    results_summary = {}

    # Determine pending variants
    variant_names = list(all_vectors.keys())
    pending_variants = ckpt.get_pending(variant_names)

    # Load cached results for completed variants
    for variant_name in variant_names:
        if ckpt.is_complete(variant_name):
            cached = ckpt.get_result(variant_name)
            if cached:
                results_summary[variant_name] = cached

    if pending_variants:
        LOG.info("Generating for variants: %s (skipping completed: %s)",
                 pending_variants, [v for v in variant_names if v not in pending_variants])
    else:
        LOG.info("All variants already complete (resuming from checkpoint)")
        return {"variants": variant_names, "results": results_summary}

    for variant_name in pending_variants:
        vector = all_vectors[variant_name]
        LOG.info("Generating for variant: %s", variant_name)
        scale = 0.0 if vector is None else 1.0

        # Create variant output directory
        variant_dir = ctx.responses_dir / variant_name
        variant_dir.mkdir(exist_ok=True)

        # Multiple choice evaluation
        mc_result = evaluate_multiple_choice(
            model, loaded.tokenizer, mc_items,
            layer_index=layer_index,
            steering_vector=vector,
            scale=scale,
            max_length=steering_cfg.get("max_length", 512),
            primary_device=device,
            seed=seed,
        )

        with open(variant_dir / "mc.json", "w") as f:
            json.dump({
                "stats": {
                    "accuracy": mc_result["stats"].accuracy,
                    "avg_correct_prob": mc_result["stats"].avg_correct_prob,
                    "total": mc_result["stats"].total,
                },
                "details": mc_result["details"],
            }, f, indent=2)

        # Generation evaluation (without judge - that's next stage)
        gen_result = evaluate_generation(
            model, loaded.tokenizer, test_items,
            layer_index=layer_index,
            steering_vector=vector,
            scale=scale,
            generation_cfg=gen_cfg,
            primary_device=device,
            judge=None,
            semantic_judge=None,
        )

        with open(variant_dir / "generation.json", "w") as f:
            json.dump({
                "stats": {
                    "total": gen_result["stats"].total,
                },
                "details": gen_result["details"],
            }, f, indent=2)

        variant_result = {
            "mc_accuracy": mc_result["stats"].accuracy,
            "mc_total": mc_result["stats"].total,
            "gen_total": gen_result["stats"].total,
        }
        results_summary[variant_name] = variant_result

        # Checkpoint this variant's completion
        ckpt.mark_complete(variant_name, variant_result)

        LOG.info("  MC accuracy: %.3f, Gen samples: %d (checkpointed)",
                 mc_result["stats"].accuracy, gen_result["stats"].total)

    return {
        "variants": variant_names,
        "results": results_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate responses with steering")
    parser.add_argument("--model", required=True, help="Model config name")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--variants", nargs="+", help="Specific variants to run")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if complete")
    args = parser.parse_args()

    config = load_config(args.model, args.config_dir)
    setup_environment(config)

    ctx = get_or_create_run(args.model, config, args.run_id)
    setup_logging(args.verbose, ctx.run_dir / "logs" / "generate.log")

    LOG.info("Run directory: %s", ctx.run_dir)

    # Check prerequisites
    if not check_stage_complete(ctx, "extract"):
        LOG.error("Extraction stage not complete. Run extract stage first.")
        return 1

    # Training is optional - we can still run baseline and steered
    if not check_stage_complete(ctx, "train"):
        LOG.warning("Training stage not complete - MLP variants will be skipped")

    if check_stage_complete(ctx, "generate") and not args.force:
        LOG.info("Generation already complete, skipping (use --force to re-run)")
        return 0

    try:
        metadata = run_generation(ctx, args.variants, force=args.force)
        save_stage_metadata(ctx, "generate", metadata)
        LOG.info("Generation stage complete")
        return 0
    except Exception as e:
        LOG.exception("Generation failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
