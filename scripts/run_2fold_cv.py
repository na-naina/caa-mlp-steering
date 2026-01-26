#!/usr/bin/env python3
"""
Run 2-fold cross-validation for TruthfulQA evaluation.

This matches the methodology used by RaLFiT, ITI, and TruthX:
- Split 817 questions into 2 folds (~408 each)
- For each fold: train on one half, test on other half
- Report average across both folds

Usage:
    python scripts/run_2fold_cv.py --model llama2_7b_chat
    python scripts/run_2fold_cv.py --model llama2_7b_chat --fold 1  # Run only fold 1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# TruthfulQA has 817 questions
TOTAL_QUESTIONS = 817
FOLD_SIZE = TOTAL_QUESTIONS // 2  # 408


def create_fold_splits(seed: int = 42):
    """Create 2-fold CV splits."""
    rng = np.random.default_rng(seed)
    indices = np.arange(TOTAL_QUESTIONS)
    rng.shuffle(indices)

    fold1_train = indices[:FOLD_SIZE].tolist()
    fold1_test = indices[FOLD_SIZE:].tolist()

    fold2_train = indices[FOLD_SIZE:].tolist()
    fold2_test = indices[:FOLD_SIZE].tolist()

    return {
        "fold1": {"train_indices": fold1_train, "test_indices": fold1_test},
        "fold2": {"train_indices": fold2_train, "test_indices": fold2_test},
    }


def create_fold_config(fold_name: str, train_indices: list, test_indices: list, output_dir: Path):
    """Create splits.json for a specific fold."""
    # Within training set, allocate to steering_pool, train, val
    n_train = len(train_indices)

    # Allocate ~408 training examples: 50 steering, 200 train, 158 val
    steering_pool_size = 50
    train_size = 200
    val_size = n_train - steering_pool_size - train_size  # ~158

    splits = {
        "steering_pool": train_indices[:steering_pool_size],
        "train": train_indices[steering_pool_size:steering_pool_size + train_size],
        "val": train_indices[steering_pool_size + train_size:],
        "test": test_indices,
    }

    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_file = output_dir / "fold_splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Created {fold_name} splits:")
    print(f"  steering_pool: {len(splits['steering_pool'])}")
    print(f"  train: {len(splits['train'])}")
    print(f"  val: {len(splits['val'])}")
    print(f"  test: {len(splits['test'])}")

    return splits_file, splits


def run_fold(model: str, fold_name: str, splits_file: Path, output_dir: Path):
    """Run the pipeline for one fold."""
    print(f"\n{'='*60}")
    print(f"Running {fold_name} for {model}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "run.py",
        "--model", model,
        "--splits-file", str(splits_file),
        "--output-dir", str(output_dir),
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode != 0:
        print(f"Error running {fold_name}")
        return False
    return True


def aggregate_results(fold_dirs: list[Path], output_dir: Path):
    """Aggregate results from both folds."""
    print(f"\n{'='*60}")
    print("Aggregating 2-fold CV results")
    print(f"{'='*60}\n")

    all_mc_results = []
    all_gen_results = []

    for fold_dir in fold_dirs:
        # MC results
        mc_file = fold_dir / "mc_proper_summary.json"
        if mc_file.exists():
            with open(mc_file) as f:
                all_mc_results.append(json.load(f))

        # Generation results (from GPT judge)
        for condition in ["baseline", "steered", "mlp_gen", "mlp_mc"]:
            gen_file = fold_dir / condition / "scale_1.00" / "gpt_judge_results.json"
            if condition == "baseline":
                gen_file = fold_dir / condition / "scale_0.00" / "gpt_judge_results.json"
            if gen_file.exists():
                with open(gen_file) as f:
                    data = json.load(f)
                    all_gen_results.append({
                        "fold": fold_dir.name,
                        "condition": condition,
                        "stats": data.get("stats", {})
                    })

    # Average MC results
    if all_mc_results:
        aggregated_mc = {}
        for method in all_mc_results[0].keys():
            mc1_scores = [r[method]["mc1_accuracy"] for r in all_mc_results if method in r]
            mc2_scores = [r[method]["mc2_score"] for r in all_mc_results if method in r]
            n_samples = sum(r[method]["n_samples"] for r in all_mc_results if method in r)

            aggregated_mc[method] = {
                "mc1_accuracy": np.mean(mc1_scores),
                "mc1_std": np.std(mc1_scores) if len(mc1_scores) > 1 else 0,
                "mc2_score": np.mean(mc2_scores),
                "mc2_std": np.std(mc2_scores) if len(mc2_scores) > 1 else 0,
                "n_samples": n_samples,
                "n_folds": len(mc1_scores),
            }

        # Save aggregated MC
        with open(output_dir / "mc_2fold_cv_results.json", "w") as f:
            json.dump(aggregated_mc, f, indent=2)

        # Print summary
        print("\n2-Fold CV MC Results:")
        print(f"{'Method':<12} {'MC1%':>8} {'MC2%':>8} {'N':>6}")
        print("-" * 40)
        for method, stats in aggregated_mc.items():
            print(f"{method:<12} {stats['mc1_accuracy']*100:>7.2f}% {stats['mc2_score']*100:>7.2f}% {stats['n_samples']:>6}")

    return aggregated_mc if all_mc_results else None


def main():
    parser = argparse.ArgumentParser(description="Run 2-fold CV evaluation")
    parser.add_argument("--model", required=True, help="Model config name")
    parser.add_argument("--fold", type=int, choices=[1, 2], help="Run only specific fold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--skip-run", action="store_true", help="Skip running, just aggregate existing results")
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path("data/outputs") / f"{args.model}_2fold_{timestamp}"
    base_output.mkdir(parents=True, exist_ok=True)

    # Create fold splits
    folds = create_fold_splits(args.seed)

    # Save fold info
    with open(base_output / "fold_info.json", "w") as f:
        json.dump({"seed": args.seed, "total_questions": TOTAL_QUESTIONS}, f, indent=2)

    fold_dirs = []

    # Run each fold
    for fold_num, (fold_name, fold_data) in enumerate(folds.items(), 1):
        if args.fold and args.fold != fold_num:
            continue

        fold_output = base_output / fold_name
        splits_file, splits = create_fold_config(
            fold_name,
            fold_data["train_indices"],
            fold_data["test_indices"],
            fold_output
        )

        if not args.skip_run:
            success = run_fold(args.model, fold_name, splits_file, fold_output)
            if not success:
                print(f"Failed to run {fold_name}")
                continue

        fold_dirs.append(fold_output)

    # Aggregate results
    if len(fold_dirs) == 2:
        aggregate_results(fold_dirs, base_output)
    else:
        print(f"\nOnly {len(fold_dirs)} fold(s) completed. Run both folds to aggregate.")

    print(f"\nResults saved to: {base_output}")


if __name__ == "__main__":
    main()
