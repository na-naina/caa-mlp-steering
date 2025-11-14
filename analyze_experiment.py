#!/usr/bin/env python3
"""
Comprehensive TruthfulQA CAA-MLP Steering Analysis
Generates metrics, visualizations, and detailed examples.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def load_results(run_dir: Path) -> dict:
    results_file = run_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    with open(results_file) as f:
        return json.load(f)

def load_generation_details(variant: str, scale: float, run_dir: Path) -> list:
    details_file = run_dir / variant / f"scale_{scale:.2f}" / "generation_details.json"
    if not details_file.exists():
        return []
    with open(details_file) as f:
        return json.load(f)

def analyze_verbosity(details: list) -> dict:
    if not details:
        return {}

    lengths = [len(d["generated"].split()) for d in details]
    has_followup = sum(1 for d in details if "\nQuestion:" in d["generated"] or "\n\n" in d["generated"])

    return {
        "avg_words": sum(lengths) / len(lengths),
        "median_words": sorted(lengths)[len(lengths)//2],
        "min_words": min(lengths),
        "max_words": max(lengths),
        "std_words": np.std(lengths),
        "pct_with_followup": 100 * has_followup / len(details),
        "word_counts": lengths,
    }

def find_interesting_examples(baseline_details, variant_details, n=10):
    """Find most interesting comparison examples."""
    examples = {
        "improvements": [],
        "degradations": [],
        "semantic_inflations": [],  # High semantic but judge disagrees
        "true_improvements": [],  # Both metrics improve
    }

    for b, v in zip(baseline_details, variant_details):
        if b["question"] != v["question"]:
            continue

        baseline_match = b.get("match", 0)
        variant_match = v.get("match", 0)
        baseline_sem = b.get("semantic_score", 0)
        variant_sem = v.get("semantic_score", 0)

        delta_match = variant_match - baseline_match
        delta_sem = variant_sem - baseline_sem

        example = {
            "question": b["question"],
            "baseline_answer": b["generated"],
            "variant_answer": v["generated"],
            "baseline_match": baseline_match,
            "variant_match": variant_match,
            "baseline_sem": baseline_sem,
            "variant_sem": variant_sem,
            "delta_match": delta_match,
            "delta_sem": delta_sem,
            "baseline_explanation": b.get("explanation", ""),
            "variant_explanation": v.get("explanation", ""),
            "correct_answers": b.get("true_answers", []),
        }

        # Categorize
        if delta_match > 0:
            examples["improvements"].append(example)
        elif delta_match < 0:
            examples["degradations"].append(example)

        if delta_sem > 0.15 and delta_match <= 0:
            examples["semantic_inflations"].append(example)

        if delta_match > 0 and delta_sem > 0:
            examples["true_improvements"].append(example)

    # Sort
    examples["improvements"].sort(key=lambda x: x["delta_sem"], reverse=True)
    examples["degradations"].sort(key=lambda x: x["delta_sem"])
    examples["semantic_inflations"].sort(key=lambda x: x["delta_sem"], reverse=True)
    examples["true_improvements"].sort(key=lambda x: x["delta_match"] + x["delta_sem"], reverse=True)

    return {k: v[:n] for k, v in examples.items()}

def plot_main_metrics(results: dict, output_dir: Path):
    """Generate main comparison plots."""
    all_variants = ["baseline", "steered", "mlp_mc", "mlp_gen"]
    all_labels = ["Baseline\n(no steering)", "Raw CAA\n(scale=1.0)", "MLP-MC", "MLP-Gen"]
    all_colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    # Filter to only variants that exist in results
    variants = []
    labels = []
    colors = []
    for var, label, color in zip(all_variants, all_labels, all_colors):
        if var in results:
            variants.append(var)
            labels.append(label)
            colors.append(color)

    # Extract metrics
    accuracies = []
    semantics = []
    for variant in variants:
        scale_key = "scale_0.00" if variant == "baseline" else "scale_1.00"
        gen = results[variant][scale_key]["generation"]
        # Handle both old and new structure
        if "stats" in gen:
            accuracies.append(gen["stats"]["accuracy"])
            semantics.append(gen["stats"]["semantic_mean"])
        else:
            accuracies.append(gen["accuracy"])
            semantics.append(gen["semantic_mean"])

    baseline_acc = accuracies[0]
    baseline_sem = semantics[0]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TruthfulQA CAA-MLP Steering Results", fontsize=16, fontweight='bold')

    # 1. Judge Accuracy
    ax = axes[0, 0]
    bars = ax.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=baseline_acc, color='green', linestyle='--', linewidth=2, label='Baseline', alpha=0.7)
    ax.set_ylabel("Judge Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Judge Accuracy by Variant", fontsize=13)
    ax.set_ylim(0, max(accuracies) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        delta = ((acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}\n({delta:+.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.legend()

    # 2. Semantic Similarity
    ax = axes[0, 1]
    bars = ax.bar(labels, semantics, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=baseline_sem, color='green', linestyle='--', linewidth=2, label='Baseline', alpha=0.7)
    ax.set_ylabel("Semantic Similarity", fontsize=12, fontweight='bold')
    ax.set_title("Semantic Similarity by Variant", fontsize=13)
    ax.set_ylim(0, max(semantics) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    for bar, sem in zip(bars, semantics):
        height = bar.get_height()
        delta = ((sem - baseline_sem) / baseline_sem * 100) if baseline_sem > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{sem:.3f}\n({delta:+.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.legend()

    # 3. Relative Change
    ax = axes[1, 0]
    delta_accs = [(a - baseline_acc) / baseline_acc * 100 for a in accuracies]
    delta_sems = [(s - baseline_sem) / baseline_sem * 100 for s in semantics]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, delta_accs, width, label='Judge Accuracy Δ',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, delta_sems, width, label='Semantic Similarity Δ',
                   color='#e74c3c', alpha=0.8, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel("Relative Change (%)", fontsize=12, fontweight='bold')
    ax.set_title("Relative Change vs Baseline", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center',
                       va='bottom' if height > 0 else 'top', fontsize=9)

    # 4. Judge vs Semantic Scatter
    ax = axes[1, 1]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.scatter(semantics[i], accuracies[i], s=300, c=color,
                  alpha=0.7, edgecolor='black', linewidth=2, label=label)
        ax.text(semantics[i] + 0.01, accuracies[i], label.split('\n')[0],
               fontsize=9, va='center')

    ax.set_xlabel("Semantic Similarity", fontsize=12, fontweight='bold')
    ax.set_ylabel("Judge Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Judge Accuracy vs Semantic Similarity", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Add diagonal line (perfect correlation)
    lims = [0, max(max(semantics), max(accuracies)) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Perfect correlation')
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "main_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'main_metrics.png'}")
    plt.close()

def plot_verbosity_analysis(baseline_details, variant_details, output_dir: Path):
    """Plot verbosity distribution."""
    baseline_verb = analyze_verbosity(baseline_details)
    variant_verb = analyze_verbosity(variant_details)

    if not baseline_verb or not variant_verb:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Word count distribution
    ax = axes[0]
    ax.hist(baseline_verb["word_counts"], bins=20, alpha=0.6, label='Baseline',
            color='green', edgecolor='black')
    ax.hist(variant_verb["word_counts"], bins=20, alpha=0.6, label='MLP-Gen',
            color='purple', edgecolor='black')
    ax.axvline(baseline_verb["avg_words"], color='green', linestyle='--', linewidth=2)
    ax.axvline(variant_verb["avg_words"], color='purple', linestyle='--', linewidth=2)
    ax.set_xlabel("Words per Answer", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax.set_title("Answer Length Distribution", fontsize=13)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Summary metrics
    ax = axes[1]
    metrics = ['Avg Words', 'Median Words', 'Follow-up %']
    baseline_vals = [baseline_verb["avg_words"], baseline_verb["median_words"],
                    baseline_verb["pct_with_followup"]]
    variant_vals = [variant_verb["avg_words"], variant_verb["median_words"],
                   variant_verb["pct_with_followup"]]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                  color='green', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, variant_vals, width, label='MLP-Gen',
                  color='purple', alpha=0.7, edgecolor='black')

    ax.set_ylabel("Value", fontsize=12, fontweight='bold')
    ax.set_title("Verbosity Metrics Comparison", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "verbosity_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'verbosity_analysis.png'}")
    plt.close()

def generate_example_report(examples: dict, output_file: Path):
    """Generate detailed example comparisons."""
    with open(output_file, 'w') as f:
        f.write("# DETAILED EXAMPLE COMPARISONS\n\n")
        f.write("=" * 80 + "\n\n")

        # True Improvements
        if examples["true_improvements"]:
            f.write("## 🎉 TRUE IMPROVEMENTS (Both Metrics Better)\n\n")
            for i, ex in enumerate(examples["true_improvements"][:5], 1):
                f.write(f"### Example {i}\n\n")
                f.write(f"**Question:** {ex['question']}\n\n")
                f.write(f"**Correct Answers:**\n")
                for ans in ex['correct_answers'][:3]:
                    f.write(f"- {ans}\n")
                f.write("\n")

                f.write(f"**Baseline:**\n```\n{ex['baseline_answer'][:200]}...\n```\n")
                f.write(f"- Judge: {'✅ CORRECT' if ex['baseline_match'] else '❌ WRONG'} (match={ex['baseline_match']})\n")
                f.write(f"- Semantic: {ex['baseline_sem']:.3f}\n")
                f.write(f"- Explanation: {ex['baseline_explanation'][:150]}...\n\n")

                f.write(f"**MLP-Gen:**\n```\n{ex['variant_answer'][:200]}...\n```\n")
                f.write(f"- Judge: {'✅ CORRECT' if ex['variant_match'] else '❌ WRONG'} (match={ex['variant_match']})\n")
                f.write(f"- Semantic: {ex['variant_sem']:.3f}\n")
                f.write(f"- Explanation: {ex['variant_explanation'][:150]}...\n\n")

                f.write(f"**Changes:** Judge {ex['delta_match']:+d}, Semantic {ex['delta_sem']:+.3f}\n\n")
                f.write("-" * 80 + "\n\n")

        # Semantic Inflations
        if examples["semantic_inflations"]:
            f.write("## ⚠️  SEMANTIC INFLATIONS (Semantic ↑ but Judge ✗)\n\n")
            for i, ex in enumerate(examples["semantic_inflations"][:5], 1):
                f.write(f"### Example {i}\n\n")
                f.write(f"**Question:** {ex['question']}\n\n")

                f.write(f"**Baseline:**\n```\n{ex['baseline_answer'][:200]}...\n```\n")
                f.write(f"- Judge: {'✅' if ex['baseline_match'] else '❌'} | Semantic: {ex['baseline_sem']:.3f}\n\n")

                f.write(f"**MLP-Gen:**\n```\n{ex['variant_answer'][:200]}...\n```\n")
                f.write(f"- Judge: {'✅' if ex['variant_match'] else '❌'} | Semantic: {ex['variant_sem']:.3f} (**+{ex['delta_sem']:.3f}**)\n\n")

                f.write(f"**Analysis:** Semantic score inflated by {ex['delta_sem']:.3f} but judge still marks wrong\n\n")
                f.write("-" * 80 + "\n\n")

        # Regular Improvements
        if examples["improvements"]:
            f.write("## ✅ ACCURACY IMPROVEMENTS\n\n")
            for i, ex in enumerate(examples["improvements"][:5], 1):
                f.write(f"### Example {i}\n\n")
                f.write(f"**Q:** {ex['question']}\n\n")
                f.write(f"Baseline: ❌ → MLP-Gen: ✅\n\n")
                f.write(f"**Baseline:** {ex['baseline_answer'][:100]}...\n\n")
                f.write(f"**MLP-Gen:** {ex['variant_answer'][:100]}...\n\n")
                f.write("-" * 80 + "\n\n")

        # Degradations
        if examples["degradations"]:
            f.write("## ❌ ACCURACY DEGRADATIONS\n\n")
            for i, ex in enumerate(examples["degradations"][:5], 1):
                f.write(f"### Example {i}\n\n")
                f.write(f"**Q:** {ex['question']}\n\n")
                f.write(f"Baseline: ✅ → MLP-Gen: ❌\n\n")
                f.write(f"**Baseline:** {ex['baseline_answer'][:100]}...\n\n")
                f.write(f"**MLP-Gen:** {ex['variant_answer'][:100]}...\n\n")
                f.write("-" * 80 + "\n\n")

    print(f"✅ Saved: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_experiment.py <run_directory>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    output_dir = run_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("TRUTHFULQA CAA-MLP ANALYSIS WITH VISUALIZATIONS")
    print("=" * 80)
    print(f"\nAnalyzing: {run_dir.name}\n")

    # Load results
    results = load_results(run_dir)
    baseline_details = load_generation_details("baseline", 0.0, run_dir)
    variant_details = load_generation_details("mlp_gen", 1.0, run_dir)

    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_main_metrics(results, output_dir)

    if baseline_details and variant_details:
        plot_verbosity_analysis(baseline_details, variant_details, output_dir)

        # Find interesting examples
        print("\n📝 Extracting interesting examples...")
        examples = find_interesting_examples(baseline_details, variant_details, n=10)
        generate_example_report(examples, output_dir / "detailed_examples.md")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80 + "\n")

    baseline_gen = results["baseline"]["scale_0.00"]["generation"]
    mlp_gen = results["mlp_gen"]["scale_1.00"]["generation"]

    # Handle both old and new structure
    if "stats" in baseline_gen:
        b_acc = baseline_gen["stats"]["accuracy"]
        b_sem = baseline_gen["stats"]["semantic_mean"]
        m_acc = mlp_gen["stats"]["accuracy"]
        m_sem = mlp_gen["stats"]["semantic_mean"]
    else:
        b_acc = baseline_gen["accuracy"]
        b_sem = baseline_gen["semantic_mean"]
        m_acc = mlp_gen["accuracy"]
        m_sem = mlp_gen["semantic_mean"]

    print(f"📊 Judge Accuracy:  {b_acc:.1%} → {m_acc:.1%} " +
          f"({(m_acc - b_acc) * 100:+.1f}%)")
    print(f"📊 Semantic Score:  {b_sem:.3f} → {m_sem:.3f} " +
          f"({(m_sem / b_sem - 1) * 100:+.1f}%)")

    if baseline_details and variant_details:
        b_verb = analyze_verbosity(baseline_details)
        v_verb = analyze_verbosity(variant_details)
        print(f"📝 Avg Words:       {b_verb['avg_words']:.1f} → {v_verb['avg_words']:.1f} words")
        print(f"📝 Follow-ups:      {b_verb['pct_with_followup']:.1f}% → {v_verb['pct_with_followup']:.1f}%")

        true_impr = len(examples.get("true_improvements", []))
        sem_infl = len(examples.get("semantic_inflations", []))
        print(f"\n✅ True improvements: {true_impr}")
        print(f"⚠️  Semantic inflations: {sem_infl}")

    print(f"\n📁 Results saved to: {output_dir}/")
    print("   - main_metrics.png")
    print("   - verbosity_analysis.png")
    print("   - detailed_examples.md")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
