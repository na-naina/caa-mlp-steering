#!/usr/bin/env python3
"""Run coherence benchmarks (ARC, HellaSwag, MMLU) with and without steering.

Compares baseline model vs steered model on general NLP benchmarks
to verify steering doesn't degrade performance.

Usage:
    python scripts/run_coherence_eval.py --vectors-dir data/outputs/vectors_L8_bn16
    python scripts/run_coherence_eval.py --vectors-dir ... --tasks arc_easy arc_challenge hellaswag
    python scripts/run_coherence_eval.py --vectors-dir ... --mmlu --limit 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.steering.mlp import SteeringMLP
from src.steering.apply import _get_decoder_layer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


def make_steering_hook(vector, scale):
    """Create a forward hook that adds steering vector to residual stream."""
    def hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steer = scale * vector.to(hidden.device, dtype=hidden.dtype)
        steer = steer.unsqueeze(0).unsqueeze(0)
        hidden = hidden + steer
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


def run_lm_eval(lm_model, tasks, limit=None, batch_size="auto"):
    """Run lm-eval-harness evaluation."""
    import lm_eval

    results = lm_eval.simple_evaluate(
        model=lm_model,
        tasks=tasks,
        limit=limit,
        batch_size=batch_size,
    )
    return results


def extract_scores(results):
    """Extract key metrics from lm-eval results."""
    scores = {}
    if "results" not in results:
        return scores
    for task, metrics in results["results"].items():
        # lm-eval uses different metric names per task
        acc = metrics.get("acc,none") or metrics.get("acc_norm,none") or metrics.get("acc", None)
        acc_norm = metrics.get("acc_norm,none")
        scores[task] = {
            "acc": acc,
            "acc_norm": acc_norm,
        }
    return scores


def main():
    parser = argparse.ArgumentParser(description="Coherence benchmarks with steering")
    parser.add_argument("--vectors-dir", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--tasks", nargs="+", default=["arc_easy", "arc_challenge", "hellaswag"])
    parser.add_argument("--mmlu", action="store_true", help="Include MMLU")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per task (for faster runs)")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/coherence"))
    args = parser.parse_args()

    tasks = list(args.tasks)
    if args.mmlu:
        tasks.append("mmlu")

    LOG.info("Tasks: %s", tasks)
    if args.limit:
        LOG.info("Limit: %d examples per task", args.limit)

    # Load model via lm-eval
    LOG.info("Loading model: %s", args.model)
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=args.model,
        dtype="bfloat16",
        batch_size=args.batch_size,
    )

    # Phase 1: Baseline evaluation
    LOG.info("=== BASELINE EVALUATION ===")
    baseline_results = run_lm_eval(lm, tasks, limit=args.limit, batch_size=args.batch_size)
    baseline_scores = extract_scores(baseline_results)

    LOG.info("Baseline results:")
    for task, scores in baseline_scores.items():
        LOG.info("  %s: acc=%.4f%s", task, scores["acc"],
                 f" acc_norm={scores['acc_norm']:.4f}" if scores["acc_norm"] else "")

    # Load steering vector
    LOG.info("Loading steering vectors from %s", args.vectors_dir)
    base_vector = torch.load(args.vectors_dir / "base_vector.pt", map_location="cpu")

    # Get model from lm-eval wrapper
    hf_model = lm.model
    param_dtype = next(hf_model.parameters()).dtype
    device = next(hf_model.parameters()).device

    mlp = SteeringMLP(input_dim=base_vector.shape[0], bottleneck_dim=args.bottleneck_dim)
    mlp.load_state_dict(
        torch.load(args.vectors_dir / "mlp_mc_state_dict.pt", map_location="cpu")
    )
    mlp.eval().to(device, dtype=param_dtype)

    with torch.no_grad():
        transformed = mlp(base_vector.to(device, dtype=param_dtype).unsqueeze(0)).squeeze(0)

    # Apply steering hook
    layer = _get_decoder_layer(hf_model, args.layer)
    hook_fn = make_steering_hook(transformed, args.scale)
    handle = layer.register_forward_hook(hook_fn)

    # Phase 2: Steered evaluation
    LOG.info("=== STEERED EVALUATION (scale=%.2f) ===", args.scale)
    steered_results = run_lm_eval(lm, tasks, limit=args.limit, batch_size=args.batch_size)
    steered_scores = extract_scores(steered_results)

    handle.remove()

    LOG.info("Steered results:")
    for task, scores in steered_scores.items():
        LOG.info("  %s: acc=%.4f%s", task, scores["acc"],
                 f" acc_norm={scores['acc_norm']:.4f}" if scores["acc_norm"] else "")

    # Comparison
    LOG.info("\n=== COMPARISON (baseline vs steered) ===")
    LOG.info("%-20s  %-12s  %-12s  %-12s", "Task", "Baseline", "Steered", "Delta")
    LOG.info("-" * 60)

    comparison = {}
    for task in baseline_scores:
        b_acc = baseline_scores[task]["acc"]
        s_acc = steered_scores.get(task, {}).get("acc", 0)
        delta = s_acc - b_acc if b_acc and s_acc else None
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        LOG.info("%-20s  %-12.4f  %-12.4f  %-12s", task, b_acc or 0, s_acc or 0, delta_str)
        comparison[task] = {
            "baseline": b_acc,
            "steered": s_acc,
            "delta": delta,
        }

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model,
        "layer": args.layer,
        "bottleneck_dim": args.bottleneck_dim,
        "scale": args.scale,
        "limit": args.limit,
        "tasks": tasks,
        "baseline": baseline_scores,
        "steered": steered_scores,
        "comparison": comparison,
    }
    out_file = args.output_dir / "coherence_results.json"
    with out_file.open("w") as f:
        json.dump(output, f, indent=2)

    LOG.info("\nResults saved to %s", out_file)


if __name__ == "__main__":
    main()
