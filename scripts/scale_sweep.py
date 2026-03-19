#!/usr/bin/env python3
"""Scale sweep: find optimal steering strength for existing MLP weights.

Phase 1: MC accuracy at many scales (fast, ~2 min per scale)
Phase 2: Generate responses for top N scales (slower, ~15 min per scale)
Phase 3: Run GPT judge on generated responses (requires OPENAI_API_KEY)

Usage:
    python scripts/scale_sweep.py --vectors-dir data/outputs/vectors_L8_bn16
    python scripts/scale_sweep.py --vectors-dir ... --phase 1   # MC only
    python scripts/scale_sweep.py --vectors-dir ... --phase 3   # Judge only (after generation)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import load_causal_model
from src.data.truthfulqa import TruthfulQADatasetManager
from src.evaluation.truthfulqa import evaluate_multiple_choice, evaluate_generation
from src.steering.mlp import SteeringMLP

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


def load_steering_vector(vectors_dir: Path, bottleneck_dim: int, device, dtype):
    """Load base vector and transform with MLP."""
    base_vector = torch.load(vectors_dir / "base_vector.pt", map_location="cpu")
    hidden_dim = base_vector.shape[0]

    mlp = SteeringMLP(input_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
    mlp.load_state_dict(
        torch.load(vectors_dir / "mlp_mc_state_dict.pt", map_location="cpu")
    )
    mlp.eval().to(device, dtype=dtype)

    with torch.no_grad():
        transformed = mlp(base_vector.to(device, dtype=dtype).unsqueeze(0)).squeeze(0)

    LOG.info(
        "Vector loaded. Norm: base=%.4f, transformed=%.4f",
        base_vector.norm().item(),
        transformed.norm().item(),
    )
    return transformed


def phase1_mc_sweep(model, tokenizer, device, transformed, mc_items, scales, output_dir):
    """Fast MC accuracy sweep across scales."""
    LOG.info("=== PHASE 1: MC Accuracy Sweep (%d scales) ===", len(scales))
    mc_results = {}

    for scale in scales:
        result = evaluate_multiple_choice(
            model, tokenizer, mc_items,
            layer_index=args.layer,
            steering_vector=transformed,
            scale=scale,
            max_length=512,
            primary_device=device,
        )
        stats = result["stats"]
        mc_results[f"{scale:.2f}"] = asdict(stats)
        LOG.info("  Scale %.2f: accuracy=%.4f (correct_prob=%.4f)",
                 scale, stats.accuracy, stats.avg_correct_prob)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mc_sweep_results.json").open("w") as f:
        json.dump(mc_results, f, indent=2)

    # Print sorted summary
    LOG.info("\n=== MC SWEEP SUMMARY ===")
    sorted_scales = sorted(mc_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for scale_str, stats in sorted_scales:
        marker = " <-- BEST" if scale_str == sorted_scales[0][0] else ""
        LOG.info("  Scale %s: accuracy=%.4f%s", scale_str, stats["accuracy"], marker)

    return mc_results


def phase2_generate(model, tokenizer, device, transformed, test_items, top_scales, output_dir):
    """Generate free-form responses for top scales."""
    LOG.info("=== PHASE 2: Generating for scales %s ===", top_scales)

    gen_cfg = {
        "preset": "qa",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_new_tokens": 64,
        "max_length": 512,
        "stop_sequences": ["\n\n", "\nQuestion:"],
    }

    for scale in top_scales:
        LOG.info("Scale %.2f: generating %d responses...", scale, len(test_items))
        gen_result = evaluate_generation(
            model, tokenizer, test_items,
            layer_index=args.layer,
            steering_vector=transformed,
            scale=scale,
            generation_cfg=gen_cfg,
            primary_device=device,
            judge=None, semantic_judge=None,
        )

        # Save in GPT judge-compatible structure
        scale_dir = output_dir / f"scale_{scale:.2f}" / "mlp_mc" / f"scale_{scale:.2f}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        with (scale_dir / "generation_details.json").open("w") as f:
            json.dump(gen_result["details"], f, indent=2)
        LOG.info("  Saved %d responses to %s", len(gen_result["details"]), scale_dir)


def phase3_judge(output_dir, workers=4):
    """Run GPT judge on generated responses."""
    LOG.info("=== PHASE 3: GPT Judge Evaluation ===")

    # Import the judge evaluation function
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_with_gpt_judge import evaluate_single_file, get_judge_models

    judge_models = get_judge_models()
    if not judge_models.get("truth") or not judge_models.get("info"):
        LOG.error("Judge models not found. Check data/truthfulqa/finetune_job_*.json files.")
        return

    gen_files = list(output_dir.rglob("generation_details.json"))
    LOG.info("Found %d generation files to evaluate", len(gen_files))

    for gen_file in gen_files:
        judge_out = gen_file.parent / "gpt_judge_results.json"
        if judge_out.exists():
            LOG.info("  Skipping %s (already evaluated)", gen_file.parent)
            continue
        LOG.info("  Evaluating %s", gen_file.parent)
        evaluate_single_file(gen_file, judge_out, judge_models, workers=workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale sweep for steering strength")
    parser.add_argument("--vectors-dir", type=Path, required=True,
                        help="Directory with base_vector.pt and mlp_mc_state_dict.pt")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5])
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/scale_sweep"))
    parser.add_argument("--top-n", type=int, default=3,
                        help="Generate responses for top N scales by MC accuracy")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Run specific phase only (1=MC, 2=generate, 3=judge)")
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load dataset with same splits as training
    dataset = TruthfulQADatasetManager(seed=args.seed)
    splits = dataset.create_pipeline_splits(
        steering_pool_size=100, train_size=309, test_size=0
    )
    test_indices = splits.test
    mc_indices = [i for i in test_indices if dataset.is_valid_mc(i)]
    mc_items = dataset.get_items(mc_indices)
    test_items = dataset.get_items(test_indices)

    if args.phase == 3:
        # Judge-only mode, no model needed
        phase3_judge(args.output_dir, workers=args.judge_workers)
        sys.exit(0)

    # Load model and vectors
    LOG.info("Loading model: %s", args.model)
    loaded = load_causal_model(args.model, dtype="bfloat16", device_map="auto")
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.primary_device
    model.eval()

    param_dtype = next(model.parameters()).dtype
    transformed = load_steering_vector(
        args.vectors_dir, args.bottleneck_dim, device, param_dtype
    )

    if args.phase is None or args.phase == 1:
        mc_results = phase1_mc_sweep(
            model, tokenizer, device, transformed, mc_items, args.scales, args.output_dir
        )

    if args.phase is None or args.phase == 2:
        # Determine top scales
        mc_file = args.output_dir / "mc_sweep_results.json"
        if mc_file.exists():
            with mc_file.open() as f:
                mc_results = json.load(f)
            sorted_scales = sorted(mc_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
            top_scales = [float(s) for s, _ in sorted_scales[:args.top_n]]
        else:
            top_scales = [1.0]  # fallback

        phase2_generate(model, tokenizer, device, transformed, test_items, top_scales, args.output_dir)

    if args.phase is None:
        # Free model memory before judge
        del model, tokenizer
        import gc; gc.collect()
        torch.cuda.empty_cache()
        phase3_judge(args.output_dir, workers=args.judge_workers)

    LOG.info("\nDone! All results in: %s", args.output_dir)
