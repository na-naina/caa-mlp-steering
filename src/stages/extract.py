#!/usr/bin/env python3
"""Stage 1: Extract steering vectors from model activations.

Resource requirements: Inference only (no gradients)
- Single GPU sufficient for most models
- ~24GB VRAM for 12B models
- ~48GB VRAM for 27B models (may need 2 GPUs)

Outputs:
- vectors/base_vector.pt
- vectors/vector_bank.pt
- metadata/extract.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from src.data.truthfulqa import TruthfulQADatasetManager
from src.models.loader import load_causal_model, _parse_dtype
from src.stages.common import (
    RunContext,
    check_stage_complete,
    get_or_create_run,
    load_config,
    save_stage_metadata,
    set_random_seeds,
    setup_environment,
    setup_logging,
)
from src.steering.extract import ActivationExtractor, compute_caa_vector
from src.steering.vector_bank import VectorBankBuilder

LOG = logging.getLogger(__name__)


def run_extraction(ctx: RunContext) -> dict:
    """Extract steering vectors from model activations."""
    config = ctx.config
    model_cfg = config["model"]
    steering_cfg = config.get("steering", {})
    seed = config.get("run", {}).get("seed", 42)

    set_random_seeds(seed)

    # Load model (inference only)
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

    # Log device allocation
    if hasattr(model, "hf_device_map"):
        devices = set(model.hf_device_map.values())
        LOG.info("Model distributed across: %s", sorted(devices))

    # Load dataset
    tqa_cfg = config.get("truthfulqa", {})
    dataset = TruthfulQADatasetManager(
        dataset_name=tqa_cfg.get("dataset_name", "truthful_qa"),
        dataset_config=tqa_cfg.get("dataset_config", "generation"),
        cache_dir=tqa_cfg.get("cache_dir"),
        seed=seed,
    )

    # Create splits
    split_cfg = tqa_cfg.get("split", {})
    splits = dataset.create_pipeline_splits(
        steering_pool_size=split_cfg.get("steering_pool", 100),
        train_size=split_cfg.get("train", 250),
        val_size=split_cfg.get("val", 117),
        test_size=split_cfg.get("test", 200),
    )

    # Save splits
    with open(ctx.metadata_dir / "splits.json", "w") as f:
        json.dump({
            "steering_pool": splits.steering_pool,
            "train": splits.train,
            "val": splits.val,
            "test": splits.test,
        }, f, indent=2)

    # Create extractor
    autocast_cfg = steering_cfg.get("autocast_dtype")
    autocast_dtype = _parse_dtype(autocast_cfg) if autocast_cfg else None

    extractor = ActivationExtractor(
        loaded,
        model_cfg["layer"],
        max_length=steering_cfg.get("max_length", 512),
        batch_size=steering_cfg.get("extract_batch_size", steering_cfg.get("batch_size", 8)),
        safe_attention=steering_cfg.get("safe_attention", False),
        autocast_dtype=autocast_dtype,
    )

    # Extract activations
    pool_pos, pool_neg, valid_prompt_indices = dataset.build_caa_prompts(splits.steering_pool)
    LOG.info("Extracting activations from %d prompt pairs", len(pool_pos))

    pos_acts, pos_valid = extractor.collect_mean_activations(pool_pos)
    neg_acts, neg_valid = extractor.collect_mean_activations(pool_neg)

    # Filter to valid pairs
    valid_pairs = sorted(set(pos_valid) & set(neg_valid))
    if len(valid_pairs) < len(pool_pos):
        LOG.warning(
            "Filtered to %d valid pairs (from %d) due to NaN/Inf",
            len(valid_pairs), len(pool_pos)
        )

    pos_mask = torch.tensor([i in valid_pairs for i in pos_valid])
    neg_mask = torch.tensor([i in valid_pairs for i in neg_valid])
    pos_acts = pos_acts[pos_mask]
    neg_acts = neg_acts[neg_mask]

    if len(pos_acts) == 0:
        raise RuntimeError("No valid activation pairs - cannot compute steering vectors")

    # Diagnostic: check activation statistics before computing vector
    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    diff = pos_mean - neg_mean
    diff_norm = diff.norm().item()

    LOG.info("Activation statistics:")
    LOG.info("  Positive mean norm: %.4e", pos_mean.norm().item())
    LOG.info("  Negative mean norm: %.4e", neg_mean.norm().item())
    LOG.info("  Difference norm (pre-normalize): %.4e", diff_norm)

    if diff_norm < 1e-6:
        LOG.error(
            "CRITICAL: Positive and negative activations are nearly identical! "
            "This suggests the model doesn't differentiate at layer %d. "
            "Try a different layer or check if prompts are correct.",
            model_cfg["layer"]
        )

    # Compute base vector (normalize=False keeps original magnitude for proper steering)
    normalize = steering_cfg.get("normalize_vector", False)
    base_vector = compute_caa_vector(pos_acts, neg_acts, normalize=normalize)
    torch.save(base_vector.cpu(), ctx.vectors_dir / "base_vector.pt")

    # Build vector bank
    bank_cfg = steering_cfg.get("vector_bank", {})
    builder = VectorBankBuilder(pos_acts, neg_acts, normalize=normalize, seed=seed)
    vector_bank = builder.build(
        num_vectors=bank_cfg.get("num_vectors", 12),
        sample_size_range=(bank_cfg.get("min_samples", 30), bank_cfg.get("max_samples", 50)),
    )

    torch.save({
        "base_vector": vector_bank.base_vector.cpu(),
        "vectors": [v.cpu() for v in vector_bank.vectors],
        "indices": vector_bank.indices,
    }, ctx.vectors_dir / "vector_bank.pt")

    # Save raw activations for debugging
    torch.save({
        "pos_acts": pos_acts.cpu(),
        "neg_acts": neg_acts.cpu(),
        "valid_pairs": valid_pairs,
    }, ctx.vectors_dir / "raw_activations.pt")

    metadata = {
        "model": model_cfg["name"],
        "layer": model_cfg["layer"],
        "num_pairs": len(pos_acts),
        "hidden_dim": pos_acts.shape[1],
        "base_vector_norm": base_vector.norm().item(),
        "diff_norm_pre_normalize": diff_norm,
        "num_bank_vectors": len(vector_bank.vectors),
    }

    LOG.info("Extraction complete: %d pairs, dim=%d, norm=%.4f",
             len(pos_acts), pos_acts.shape[1], base_vector.norm().item())

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Extract steering vectors")
    parser.add_argument("--model", required=True, help="Model config name")
    parser.add_argument("--run-id", help="Resume existing run")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if complete")
    args = parser.parse_args()

    config = load_config(args.model, args.config_dir)
    setup_environment(config)

    ctx = get_or_create_run(args.model, config, args.run_id)
    setup_logging(args.verbose, ctx.run_dir / "logs" / "extract.log")

    LOG.info("Run directory: %s", ctx.run_dir)

    if check_stage_complete(ctx, "extract") and not args.force:
        LOG.info("Extraction already complete, skipping (use --force to re-run)")
        return 0

    try:
        metadata = run_extraction(ctx)
        save_stage_metadata(ctx, "extract", metadata)
        LOG.info("Extraction stage complete")
        # Print run_id for downstream stages
        print(f"RUN_ID={ctx.run_id}")
        return 0
    except Exception as e:
        LOG.exception("Extraction failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
