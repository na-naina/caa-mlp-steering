#!/usr/bin/env python3
"""Analyze TruthfulQA results by category."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset


def load_data(results_dir: Path):
    """Load all necessary data files."""
    # Load dataset with categories
    dataset = load_dataset("truthful_qa", "generation", cache_dir="./cache")["validation"]

    # Load splits
    with open(results_dir / "metadata/splits.json") as f:
        splits = json.load(f)

    # Load generation details
    with open(results_dir / "baseline/scale_0.00/generation_details.json") as f:
        baseline_list = json.load(f)

    with open(results_dir / "steered/scale_1.00/generation_details.json") as f:
        raw_caa_list = json.load(f)

    with open(results_dir / "mlp_gen/scale_1.00/generation_details.json") as f:
        mlp_gen_list = json.load(f)

    # Create question-indexed dicts
    baseline = {item["question"]: item for item in baseline_list}
    raw_caa = {item["question"]: item for item in raw_caa_list}
    mlp_gen = {item["question"]: item for item in mlp_gen_list}

    return dataset, splits, baseline, raw_caa, mlp_gen


def map_indices_to_categories(dataset, indices):
    """Map dataset indices to their categories."""
    categories = {}
    for idx in indices:
        item = dataset[idx]
        question = item["question"]
        category = item["category"]
        categories[question] = category
    return categories


def analyze_by_category(dataset, splits, baseline, raw_caa, mlp_gen):
    """Analyze performance by category."""

    # Map categories
    steering_cats = map_indices_to_categories(dataset, splits["steering_pool"])
    test_cats = map_indices_to_categories(dataset, splits["test"])

    # Category performance tracking
    category_stats = defaultdict(lambda: {
        "baseline": {"correct": 0, "total": 0},
        "raw_caa": {"correct": 0, "total": 0},
        "mlp_gen": {"correct": 0, "total": 0},
    })

    # Analyze test set performance
    for question in baseline.keys():
        if question not in test_cats:
            continue

        category = test_cats[question]

        # Baseline
        if baseline[question]["match"] == 1:
            category_stats[category]["baseline"]["correct"] += 1
        category_stats[category]["baseline"]["total"] += 1

        # Raw CAA
        if question in raw_caa:
            if raw_caa[question]["match"] == 1:
                category_stats[category]["raw_caa"]["correct"] += 1
            category_stats[category]["raw_caa"]["total"] += 1

        # MLP-Gen
        if question in mlp_gen:
            if mlp_gen[question]["match"] == 1:
                category_stats[category]["mlp_gen"]["correct"] += 1
            category_stats[category]["mlp_gen"]["total"] += 1

    # Compute accuracies
    category_accuracies = {}
    for category, stats in category_stats.items():
        category_accuracies[category] = {
            "baseline": stats["baseline"]["correct"] / stats["baseline"]["total"] if stats["baseline"]["total"] > 0 else 0,
            "raw_caa": stats["raw_caa"]["correct"] / stats["raw_caa"]["total"] if stats["raw_caa"]["total"] > 0 else 0,
            "mlp_gen": stats["mlp_gen"]["correct"] / stats["mlp_gen"]["total"] if stats["mlp_gen"]["total"] > 0 else 0,
            "count": stats["baseline"]["total"],
        }

    return steering_cats, test_cats, category_stats, category_accuracies


def plot_category_analysis(steering_cats, test_cats, category_accuracies, output_dir):
    """Create visualizations for category analysis."""

    # Count categories
    from collections import Counter
    steering_counts = Counter(steering_cats.values())
    test_counts = Counter(test_cats.values())

    # Get categories sorted by test count
    categories_by_count = sorted(
        [(cat, count) for cat, count in test_counts.items() if count >= 2],
        key=lambda x: x[1],
        reverse=True
    )[:15]  # Top 15 categories

    categories = [cat for cat, _ in categories_by_count]

    # Prepare data
    baseline_acc = [category_accuracies[cat]["baseline"] * 100 for cat in categories]
    raw_caa_acc = [category_accuracies[cat]["raw_caa"] * 100 for cat in categories]
    mlp_gen_acc = [category_accuracies[cat]["mlp_gen"] * 100 for cat in categories]
    counts = [category_accuracies[cat]["count"] for cat in categories]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: Accuracy by category
    x = np.arange(len(categories))
    width = 0.25

    ax1.barh(x - width, baseline_acc, width, label='Baseline', color='#2ecc71', alpha=0.8)
    ax1.barh(x, raw_caa_acc, width, label='Raw CAA', color='#3498db', alpha=0.8)
    ax1.barh(x + width, mlp_gen_acc, width, label='MLP-Gen', color='#9b59b6', alpha=0.8)

    ax1.set_yticks(x)
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.set_xlabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Accuracy by Category (Top 15)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 100)

    # Add count labels
    for i, count in enumerate(counts):
        ax1.text(102, i, f'n={count}', va='center', fontsize=8, color='gray')

    # Plot 2: Performance delta (MLP-Gen vs Raw CAA)
    delta = [mlp_gen_acc[i] - raw_caa_acc[i] for i in range(len(categories))]
    colors = ['#27ae60' if d > 0 else '#e74c3c' for d in delta]

    ax2.barh(x, delta, color=colors, alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(categories, fontsize=9)
    ax2.set_xlabel('Accuracy Delta (%)', fontsize=11)
    ax2.set_title('MLP-Gen vs Raw CAA (by Category)', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "category_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'category_analysis.png'}")
    plt.close()

    # Plot 3: Category distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Steering pool
    top_steering = sorted(steering_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    categories_s, counts_s = zip(*top_steering)
    ax1.barh(range(len(categories_s)), counts_s, color='#3498db', alpha=0.8)
    ax1.set_yticks(range(len(categories_s)))
    ax1.set_yticklabels(categories_s, fontsize=10)
    ax1.set_xlabel('Count', fontsize=11)
    ax1.set_title('Steering Pool Categories (Top 10)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Test set
    top_test = sorted(test_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    categories_t, counts_t = zip(*top_test)
    ax2.barh(range(len(categories_t)), counts_t, color='#9b59b6', alpha=0.8)
    ax2.set_yticks(range(len(categories_t)))
    ax2.set_yticklabels(categories_t, fontsize=10)
    ax2.set_xlabel('Count', fontsize=11)
    ax2.set_title('Test Set Categories (Top 10)', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "category_distribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'category_distribution.png'}")
    plt.close()


def generate_category_report(steering_cats, test_cats, category_accuracies, output_dir):
    """Generate detailed text report."""

    report = []
    report.append("# CATEGORY ANALYSIS")
    report.append("=" * 80)
    report.append("")

    # Steering pool categories
    from collections import Counter
    steering_counts = Counter(steering_cats.values())

    report.append("## STEERING POOL CATEGORIES")
    report.append("")
    report.append(f"Total steering examples: {len(steering_cats)}")
    report.append(f"Unique categories: {len(steering_counts)}")
    report.append("")
    report.append("Top 10 categories:")
    for cat, count in steering_counts.most_common(10):
        report.append(f"  {cat:30s} {count:3d} ({count/len(steering_cats)*100:.1f}%)")
    report.append("")

    # Test set categories
    test_counts = Counter(test_cats.values())
    report.append("## TEST SET CATEGORIES")
    report.append("")
    report.append(f"Total test examples: {len(test_cats)}")
    report.append(f"Unique categories: {len(test_counts)}")
    report.append("")
    report.append("Top 10 categories:")
    for cat, count in test_counts.most_common(10):
        report.append(f"  {cat:30s} {count:3d} ({count/len(test_cats)*100:.1f}%)")
    report.append("")

    # Performance by category
    report.append("## PERFORMANCE BY CATEGORY")
    report.append("")
    report.append("Categories with at least 2 test examples:")
    report.append("")
    report.append(f"{'Category':<30} {'Count':>5} {'Baseline':>9} {'Raw CAA':>9} {'MLP-Gen':>9} {'MLP Δ':>8}")
    report.append("-" * 85)

    # Sort by count
    sorted_cats = sorted(
        [(cat, acc) for cat, acc in category_accuracies.items() if acc["count"] >= 2],
        key=lambda x: x[1]["count"],
        reverse=True
    )

    for cat, acc in sorted_cats:
        baseline_acc = acc["baseline"] * 100
        raw_caa_acc = acc["raw_caa"] * 100
        mlp_gen_acc = acc["mlp_gen"] * 100
        delta = mlp_gen_acc - raw_caa_acc

        report.append(
            f"{cat:<30} {acc['count']:>5} "
            f"{baseline_acc:>8.1f}% {raw_caa_acc:>8.1f}% {mlp_gen_acc:>8.1f}% "
            f"{delta:>+7.1f}%"
        )

    report.append("")
    report.append("")

    # Best/worst categories for each variant
    report.append("## BEST CATEGORIES BY VARIANT")
    report.append("")

    # Filter categories with at least 3 examples
    filtered_cats = [(cat, acc) for cat, acc in category_accuracies.items() if acc["count"] >= 3]

    # Raw CAA best
    report.append("### Raw CAA (Top 5 Best)")
    best_caa = sorted(filtered_cats, key=lambda x: x[1]["raw_caa"], reverse=True)[:5]
    for cat, acc in best_caa:
        report.append(f"  {cat:30s} {acc['raw_caa']*100:>5.1f}% (n={acc['count']})")
    report.append("")

    # MLP-Gen best
    report.append("### MLP-Gen (Top 5 Best)")
    best_mlp = sorted(filtered_cats, key=lambda x: x[1]["mlp_gen"], reverse=True)[:5]
    for cat, acc in best_mlp:
        report.append(f"  {cat:30s} {acc['mlp_gen']*100:>5.1f}% (n={acc['count']})")
    report.append("")

    # MLP-Gen biggest improvements over Raw CAA
    report.append("### MLP-Gen Biggest Improvements vs Raw CAA (Top 5)")
    improvements = sorted(
        filtered_cats,
        key=lambda x: x[1]["mlp_gen"] - x[1]["raw_caa"],
        reverse=True
    )[:5]
    for cat, acc in improvements:
        delta = (acc["mlp_gen"] - acc["raw_caa"]) * 100
        report.append(
            f"  {cat:30s} {delta:>+6.1f}% "
            f"(CAA: {acc['raw_caa']*100:.1f}% → MLP: {acc['mlp_gen']*100:.1f}%, n={acc['count']})"
        )
    report.append("")

    # MLP-Gen biggest degradations vs Raw CAA
    report.append("### MLP-Gen Biggest Degradations vs Raw CAA (Top 5)")
    degradations = sorted(
        filtered_cats,
        key=lambda x: x[1]["mlp_gen"] - x[1]["raw_caa"]
    )[:5]
    for cat, acc in degradations:
        delta = (acc["mlp_gen"] - acc["raw_caa"]) * 100
        report.append(
            f"  {cat:30s} {delta:>+6.1f}% "
            f"(CAA: {acc['raw_caa']*100:.1f}% → MLP: {acc['mlp_gen']*100:.1f}%, n={acc['count']})"
        )
    report.append("")

    # Write report
    report_text = "\n".join(report)
    with open(output_dir / "category_analysis.txt", "w") as f:
        f.write(report_text)

    print(f"Saved: {output_dir / 'category_analysis.txt'}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="Analyze TruthfulQA results by category")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("analysis_results/run_20251105_190137"),
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save analysis outputs (defaults to results-dir/analysis)"
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    dataset, splits, baseline, raw_caa, mlp_gen = load_data(results_dir)

    print("Analyzing by category...")
    steering_cats, test_cats, category_stats, category_accuracies = analyze_by_category(
        dataset, splits, baseline, raw_caa, mlp_gen
    )

    print("Generating visualizations...")
    plot_category_analysis(steering_cats, test_cats, category_accuracies, output_dir)

    print("\nGenerating report...")
    generate_category_report(steering_cats, test_cats, category_accuracies, output_dir)

    print("\n✅ Category analysis complete!")


if __name__ == "__main__":
    main()
