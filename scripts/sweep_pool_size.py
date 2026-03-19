#!/usr/bin/env python3
"""Sweep steering pool size to find how many examples are needed.

Tests: 5, 10, 25, 50, 100 steering pool examples.
For each, runs extract → train → MC eval (fast, no generation).

Usage:
    python scripts/sweep_pool_size.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import load_causal_model
from src.data.truthfulqa import TruthfulQADatasetManager
from src.steering.extract import ActivationExtractor, compute_caa_vector
from src.steering.vector_bank import VectorBankBuilder
from src.steering.mlp import SteeringMLP
from src.steering.training import MCTrainingConfig, train_mc_mlp
from src.evaluation.truthfulqa import evaluate_multiple_choice

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/pool_sweep"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load model once
    LOG.info("Loading model: %s", args.model)
    loaded = load_causal_model(args.model, dtype="bfloat16", device_map="auto")
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.primary_device
    model.eval()
    param_dtype = next(model.parameters()).dtype

    # Load full dataset
    dataset = TruthfulQADatasetManager(seed=args.seed)

    results = {}

    for pool_size in args.pool_sizes:
        LOG.info("\n=== POOL SIZE: %d ===", pool_size)

        # Create splits with this pool size
        # Remaining goes to train + test
        train_size = min(309, 817 - pool_size - 100)  # keep at least 100 for test
        splits = dataset.create_pipeline_splits(
            steering_pool_size=pool_size,
            train_size=train_size,
            test_size=0,
        )

        # Extract
        extractor = ActivationExtractor(
            loaded, args.layer, max_length=512, batch_size=8,
        )
        pool_pos, pool_neg, _ = dataset.build_caa_prompts(splits.steering_pool)
        pos_acts, pos_valid = extractor.collect_mean_activations(pool_pos)
        neg_acts, neg_valid = extractor.collect_mean_activations(pool_neg)
        valid_pairs = sorted(set(pos_valid) & set(neg_valid))
        pos_mask = torch.tensor([i in valid_pairs for i in pos_valid])
        neg_mask = torch.tensor([i in valid_pairs for i in neg_valid])
        pos_acts, neg_acts = pos_acts[pos_mask], neg_acts[neg_mask]

        base_vector = compute_caa_vector(pos_acts, neg_acts, normalize=False)

        # Build vector bank (fewer vectors for small pools)
        num_bank = min(12, max(2, pool_size // 5))
        min_samples = min(max(3, pool_size // 4), len(valid_pairs))
        max_samples = min(max(5, pool_size // 2), len(valid_pairs))
        builder = VectorBankBuilder(pos_acts, neg_acts, normalize=False, seed=args.seed)
        vector_bank = builder.build(num_vectors=num_bank, sample_size_range=(min_samples, max_samples))

        # Train MLP
        hidden_dim = base_vector.shape[0]
        mlp = SteeringMLP(input_dim=hidden_dim, bottleneck_dim=args.bottleneck_dim).to(device, dtype=param_dtype)
        mc_cfg = MCTrainingConfig(lr=5e-4, epochs=2, steps_per_epoch=50, batch_size=4)
        train_mc_mlp(
            mlp, model=model, tokenizer=tokenizer, dataset=dataset,
            train_indices=splits.train, vector_bank=vector_bank,
            layer_index=args.layer, primary_device=device,
            max_length=512, config=mc_cfg, seed=args.seed + pool_size,
        )

        # Evaluate MC
        transformed = mlp(vector_bank.base_vector.to(device, dtype=param_dtype).unsqueeze(0)).squeeze(0).detach()
        mc_indices = [i for i in splits.test if dataset.is_valid_mc(i)]
        mc_items = dataset.get_items(mc_indices)

        mc_result = evaluate_multiple_choice(
            model, tokenizer, mc_items,
            layer_index=args.layer, steering_vector=transformed, scale=1.0,
            max_length=512, primary_device=device,
        )

        # Also eval baseline (no steering) on same test set
        baseline_result = evaluate_multiple_choice(
            model, tokenizer, mc_items,
            layer_index=args.layer, steering_vector=None, scale=0.0,
            max_length=512, primary_device=device,
        )

        results[pool_size] = {
            "pool_size": pool_size,
            "valid_pairs": len(valid_pairs),
            "bank_vectors": num_bank,
            "train_size": len(splits.train),
            "test_size": len(splits.test),
            "mc_accuracy": mc_result["stats"].accuracy,
            "baseline_mc_accuracy": baseline_result["stats"].accuracy,
            "improvement": mc_result["stats"].accuracy - baseline_result["stats"].accuracy,
        }
        LOG.info("Pool %d: MC=%.4f (baseline=%.4f, +%.4f)",
                 pool_size, mc_result["stats"].accuracy,
                 baseline_result["stats"].accuracy,
                 mc_result["stats"].accuracy - baseline_result["stats"].accuracy)

    # Summary
    LOG.info("\n=== POOL SIZE SWEEP RESULTS ===")
    LOG.info("%-10s %-12s %-12s %-12s", "Pool", "MC Acc", "Baseline", "Improvement")
    for ps in sorted(results.keys()):
        r = results[ps]
        LOG.info("%-10d %-12.4f %-12.4f %-12.4f", ps, r["mc_accuracy"], r["baseline_mc_accuracy"], r["improvement"])

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pool_sweep_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    LOG.info("Results saved to %s", args.output_dir / "pool_sweep_results.json")


if __name__ == "__main__":
    main()
