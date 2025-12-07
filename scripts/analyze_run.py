#!/usr/bin/env python3
"""Analyze results from a completed pipeline run.

Usage:
    python scripts/analyze_run.py outputs/gemma3/gemma3_4b_it_20251201_120000
    python scripts/analyze_run.py --model gemma3_4b_it --latest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_latest_run(output_root: Path, model_pattern: str) -> Path | None:
    """Find most recent run matching pattern."""
    candidates = []
    for family_dir in output_root.iterdir():
        if not family_dir.is_dir():
            continue
        for run_dir in family_dir.iterdir():
            if model_pattern in run_dir.name:
                candidates.append(run_dir)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_run(run_dir: Path) -> None:
    """Print analysis of a pipeline run."""
    print(f"\n{'=' * 60}")
    print(f"Run: {run_dir.name}")
    print(f"Path: {run_dir}")
    print(f"{'=' * 60}\n")

    # Check stage completion
    metadata_dir = run_dir / "metadata"
    stages = ["extract", "train", "generate", "score"]
    print("Stage Status:")
    for stage in stages:
        meta = load_json(metadata_dir / f"{stage}.json")
        if meta:
            print(f"  {stage}: ✓ completed at {meta.get('timestamp', 'unknown')}")
        else:
            print(f"  {stage}: ✗ not completed")
    print()

    # Extraction info
    extract_meta = load_json(metadata_dir / "extract.json")
    if extract_meta:
        print("Extraction:")
        print(f"  Model: {extract_meta.get('model')}")
        print(f"  Layer: {extract_meta.get('layer')}")
        print(f"  Pairs: {extract_meta.get('num_pairs')}")
        print(f"  Hidden dim: {extract_meta.get('hidden_dim')}")
        print(f"  Vector norm: {extract_meta.get('base_vector_norm', 0):.4f}")
        diff_norm = extract_meta.get('diff_norm_pre_normalize', 0)
        if diff_norm < 1e-6:
            print(f"  ⚠️  WARNING: Pre-normalize diff norm very low ({diff_norm:.2e})")
            print(f"     This indicates pos/neg activations are nearly identical.")
        print()

    # Training info
    train_meta = load_json(metadata_dir / "train.json")
    if train_meta:
        print("Training:")
        print(f"  MC MLP valid: {train_meta.get('mc_valid')}")
        print(f"  Gen MLP valid: {train_meta.get('gen_valid')}")
        if train_meta.get('mc_final_loss'):
            print(f"  MC final loss: {train_meta['mc_final_loss']:.4f}")
        if train_meta.get('mc_final_acc'):
            print(f"  MC final acc: {train_meta['mc_final_acc']:.3f}")
        if train_meta.get('gen_final_loss'):
            print(f"  Gen final loss: {train_meta['gen_final_loss']:.4f}")
        print()

    # Generation info
    gen_meta = load_json(metadata_dir / "generate.json")
    if gen_meta:
        print("Generation:")
        print(f"  Variants: {gen_meta.get('variants')}")
        results = gen_meta.get('results', {})
        for variant, stats in results.items():
            print(f"  {variant}:")
            print(f"    MC accuracy: {stats.get('mc_accuracy', 0):.3f}")
        print()

    # Scoring results
    summary = load_json(run_dir / "scores" / "summary.json")
    if summary:
        print("Scores:")
        for variant, stats in summary.items():
            print(f"\n  {variant}:")
            if stats.get('judge_true_rate') is not None:
                print(f"    Truth rate: {stats['judge_true_rate']:.3f}")
            if stats.get('informativeness_rate') is not None:
                print(f"    Info rate: {stats['informativeness_rate']:.3f}")
            if stats.get('semantic_true_rate') is not None:
                print(f"    Semantic match: {stats['semantic_true_rate']:.3f}")
            if stats.get('avg_length') is not None:
                print(f"    Avg length: {stats['avg_length']:.1f}")
        print()

    # Comparison table
    if summary and len(summary) > 1:
        print("\nComparison Table:")
        print(f"{'Variant':<12} {'MC Acc':>8} {'Truth':>8} {'Info':>8}")
        print("-" * 40)

        mc_results = gen_meta.get('results', {}) if gen_meta else {}
        for variant in summary:
            mc_acc = mc_results.get(variant, {}).get('mc_accuracy', 0)
            truth = summary[variant].get('judge_true_rate', 0) or 0
            info = summary[variant].get('informativeness_rate', 0) or 0
            print(f"{variant:<12} {mc_acc:>8.3f} {truth:>8.3f} {info:>8.3f}")

        # Compute deltas from baseline
        if 'baseline' in summary:
            print()
            print("Delta from baseline:")
            baseline_mc = mc_results.get('baseline', {}).get('mc_accuracy', 0)
            baseline_truth = summary['baseline'].get('judge_true_rate', 0) or 0

            for variant in summary:
                if variant == 'baseline':
                    continue
                mc_acc = mc_results.get(variant, {}).get('mc_accuracy', 0)
                truth = summary[variant].get('judge_true_rate', 0) or 0
                mc_delta = mc_acc - baseline_mc
                truth_delta = truth - baseline_truth
                sign_mc = '+' if mc_delta >= 0 else ''
                sign_truth = '+' if truth_delta >= 0 else ''
                print(f"  {variant}: MC {sign_mc}{mc_delta:.3f}, Truth {sign_truth}{truth_delta:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pipeline run results")
    parser.add_argument("run_dir", type=Path, nargs="?", help="Run directory")
    parser.add_argument("--model", help="Model pattern to find latest run")
    parser.add_argument("--latest", action="store_true", help="Find latest run")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    elif args.model:
        run_dir = find_latest_run(args.output_root, args.model)
        if not run_dir:
            print(f"No runs found matching '{args.model}'")
            return 1
    else:
        print("Specify run_dir or --model")
        return 1

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return 1

    analyze_run(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
