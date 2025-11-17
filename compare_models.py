#!/usr/bin/env python3
"""
Cross-model comparison analysis for CAA steering experiments.
Compares performance across different model sizes and families.
"""

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def parse_model_name(dirname: str) -> dict:
    """Extract model family and size from directory name."""
    # Try matching XB format (e.g., 1b, 4b, 12b, 27b)
    match = re.search(r'(gemma[23])_(\d+)b', dirname)
    if match:
        family = match.group(1)
        size_str = match.group(2)
        size = int(size_str)
        return {
            'family': family,
            'size': size,
            'size_str': f"{size_str}B",
            'full_name': f"{family.capitalize()}-{size_str}B"
        }

    # Try matching XXXm format (e.g., 270m)
    match = re.search(r'(gemma[23])_(\d+)m', dirname)
    if match:
        family = match.group(1)
        size_m = int(match.group(2))
        size = size_m / 1000.0  # Convert to billions for consistent sorting
        return {
            'family': family,
            'size': size,
            'size_str': f"{size_m}M",
            'full_name': f"{family.capitalize()}-{size_m}M"
        }

    return None


def load_all_results(base_dir: Path, model_type: str = 'pt') -> list:
    """Load results from all model runs.

    Args:
        base_dir: Base directory containing experiment outputs
        model_type: 'pt' for pretrained models, 'it' for instruct-tuned models
    """
    results = []

    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Filter based on model type
        is_it_model = '_it_2025' in run_dir.name
        if model_type == 'pt' and is_it_model:
            print(f"⚠️  Skipping {run_dir.name}: IT variant (using PT only)")
            continue
        elif model_type == 'it' and not is_it_model:
            print(f"⚠️  Skipping {run_dir.name}: PT variant (using IT only)")
            continue

        model_info = parse_model_name(run_dir.name)
        if not model_info:
            continue

        results_file = run_dir / "results.json"
        if not results_file.exists():
            print(f"⚠️  Skipping {run_dir.name}: no results.json")
            continue

        with results_file.open() as f:
            data = json.load(f)

        # Extract metrics for each variant
        model_results = {
            'model_info': model_info,
            'run_dir': run_dir,
            'variants': {}
        }

        for variant in ['baseline', 'steered', 'mlp_mc', 'mlp_gen']:
            if variant not in data:
                continue

            scale_key = "scale_0.00" if variant == "baseline" else "scale_1.00"
            if scale_key not in data[variant]:
                continue

            gen = data[variant][scale_key]['generation']

            # Handle both old and new structure
            if "stats" in gen:
                accuracy = gen["stats"]["accuracy"]
                semantic_mean = gen["stats"]["semantic_mean"]
            else:
                accuracy = gen["accuracy"]
                semantic_mean = gen["semantic_mean"]

            # Skip variants with invalid data
            if accuracy is None or semantic_mean is None:
                continue

            model_results['variants'][variant] = {
                'accuracy': accuracy,
                'semantic_mean': semantic_mean,
            }

        # Only include models that have at least baseline results
        if 'baseline' in model_results['variants']:
            results.append(model_results)
        else:
            print(f"⚠️  Skipping {run_dir.name}: no valid baseline results")

    # Sort by family and size
    results.sort(key=lambda x: (x['model_info']['family'], x['model_info']['size']))

    return results


def plot_scaling_analysis(results: list, output_dir: Path):
    """Plot how performance scales with model size."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Scaling Across Model Sizes', fontsize=16, fontweight='bold')

    # Separate by family
    gemma2_results = [r for r in results if r['model_info']['family'] == 'gemma2']
    gemma3_results = [r for r in results if r['model_info']['family'] == 'gemma3']

    # Colors for variants
    colors = {
        'baseline': '#2ecc71',
        'steered': '#3498db',
        'mlp_mc': '#e74c3c',
        'mlp_gen': '#9b59b6'
    }

    labels = {
        'baseline': 'Baseline',
        'steered': 'Raw CAA',
        'mlp_mc': 'MLP-MC',
        'mlp_gen': 'MLP-Gen'
    }

    # Plot 1: Gemma2 Accuracy Scaling
    ax = axes[0, 0]
    for variant in ['baseline', 'steered', 'mlp_mc', 'mlp_gen']:
        sizes = []
        accuracies = []
        for r in gemma2_results:
            if variant in r['variants']:
                sizes.append(r['model_info']['size'])
                accuracies.append(r['variants'][variant]['accuracy'] * 100)
        if sizes:
            ax.plot(sizes, accuracies, 'o-', label=labels.get(variant, variant),
                   color=colors.get(variant), linewidth=2, markersize=8)

    ax.set_xlabel('Model Size (B parameters)', fontsize=12)
    ax.set_ylabel('Judge Accuracy (%)', fontsize=12)
    ax.set_title('Gemma2 Family: Accuracy Scaling', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks([2, 9, 27])
    ax.set_xticklabels(['2B', '9B', '27B'])

    # Plot 2: Gemma3 Accuracy Scaling
    ax = axes[0, 1]
    for variant in ['baseline', 'steered', 'mlp_mc', 'mlp_gen']:
        sizes = []
        accuracies = []
        for r in gemma3_results:
            if variant in r['variants']:
                sizes.append(r['model_info']['size'])
                accuracies.append(r['variants'][variant]['accuracy'] * 100)
        if sizes:
            ax.plot(sizes, accuracies, 'o-', label=labels.get(variant, variant),
                   color=colors.get(variant), linewidth=2, markersize=8)

    ax.set_xlabel('Model Size (B parameters)', fontsize=12)
    ax.set_ylabel('Judge Accuracy (%)', fontsize=12)
    ax.set_title('Gemma3 Family: Accuracy Scaling', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if len(gemma3_results) > 1:
        ax.set_xscale('log')
        ax.set_xticks([r['model_info']['size'] for r in gemma3_results])
        ax.set_xticklabels([r['model_info']['size_str'] for r in gemma3_results])

    # Plot 3: Gemma2 Semantic Scaling
    ax = axes[1, 0]
    for variant in ['baseline', 'steered', 'mlp_mc', 'mlp_gen']:
        sizes = []
        semantics = []
        for r in gemma2_results:
            if variant in r['variants']:
                sizes.append(r['model_info']['size'])
                semantics.append(r['variants'][variant]['semantic_mean'])
        if sizes:
            ax.plot(sizes, semantics, 'o-', label=labels.get(variant, variant),
                   color=colors.get(variant), linewidth=2, markersize=8)

    ax.set_xlabel('Model Size (B parameters)', fontsize=12)
    ax.set_ylabel('Semantic Similarity Score', fontsize=12)
    ax.set_title('Gemma2 Family: Semantic Score Scaling', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks([2, 9, 27])
    ax.set_xticklabels(['2B', '9B', '27B'])

    # Plot 4: Gemma3 Semantic Scaling
    ax = axes[1, 1]
    for variant in ['baseline', 'steered', 'mlp_mc', 'mlp_gen']:
        sizes = []
        semantics = []
        for r in gemma3_results:
            if variant in r['variants']:
                sizes.append(r['model_info']['size'])
                semantics.append(r['variants'][variant]['semantic_mean'])
        if sizes:
            ax.plot(sizes, semantics, 'o-', label=labels.get(variant, variant),
                   color=colors.get(variant), linewidth=2, markersize=8)

    ax.set_xlabel('Model Size (B parameters)', fontsize=12)
    ax.set_ylabel('Semantic Similarity Score', fontsize=12)
    ax.set_title('Gemma3 Family: Semantic Score Scaling', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if len(gemma3_results) > 1:
        ax.set_xscale('log')
        ax.set_xticks([r['model_info']['size'] for r in gemma3_results])
        ax.set_xticklabels([r['model_info']['size_str'] for r in gemma3_results])

    plt.tight_layout()
    output_file = output_dir / "model_scaling_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_steering_effectiveness(results: list, output_dir: Path):
    """Plot steering effectiveness across models - ALL VARIANTS."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Steering Effectiveness Across Models - All Variants', fontsize=16, fontweight='bold')

    model_names = []
    baseline_accs = []
    steered_accs = []
    mlp_mc_accs = []
    mlp_gen_accs = []

    for r in results:
        if 'baseline' in r['variants']:
            model_names.append(r['model_info']['full_name'])
            baseline_accs.append(r['variants']['baseline']['accuracy'] * 100)
            steered_accs.append(r['variants']['steered']['accuracy'] * 100 if 'steered' in r['variants'] else None)
            mlp_mc_accs.append(r['variants']['mlp_mc']['accuracy'] * 100 if 'mlp_mc' in r['variants'] else None)
            mlp_gen_accs.append(r['variants']['mlp_gen']['accuracy'] * 100 if 'mlp_gen' in r['variants'] else None)

    # Plot 1: Absolute accuracies for all variants
    ax = axes[0]
    x = np.arange(len(model_names))
    width = 0.2

    ax.bar(x - 1.5*width, baseline_accs, width, label='Baseline', color='#2ecc71', alpha=0.8)

    if any(v is not None for v in steered_accs):
        ax.bar(x - 0.5*width, [v if v is not None else 0 for v in steered_accs], width,
               label='Steered (Raw CAA)', color='#3498db', alpha=0.8)

    if any(v is not None for v in mlp_mc_accs):
        ax.bar(x + 0.5*width, [v if v is not None else 0 for v in mlp_mc_accs], width,
               label='MLP-MC', color='#e74c3c', alpha=0.8)

    if any(v is not None for v in mlp_gen_accs):
        ax.bar(x + 1.5*width, [v if v is not None else 0 for v in mlp_gen_accs], width,
               label='MLP-Gen', color='#9b59b6', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Judge Accuracy (%)', fontsize=12)
    ax.set_title('All Variants Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Improvements over baseline for each variant
    ax = axes[1]
    steered_improvements = [(s - b) if s is not None else None for b, s in zip(baseline_accs, steered_accs)]
    mlp_mc_improvements = [(m - b) if m is not None else None for b, m in zip(baseline_accs, mlp_mc_accs)]
    mlp_gen_improvements = [(m - b) if m is not None else None for b, m in zip(baseline_accs, mlp_gen_accs)]

    if any(v is not None for v in steered_improvements):
        ax.bar(x - width, [v if v is not None else 0 for v in steered_improvements], width,
               label='Steered Δ', color='#3498db', alpha=0.8)

    if any(v is not None for v in mlp_mc_improvements):
        ax.bar(x, [v if v is not None else 0 for v in mlp_mc_improvements], width,
               label='MLP-MC Δ', color='#e74c3c', alpha=0.8)

    if any(v is not None for v in mlp_gen_improvements):
        ax.bar(x + width, [v if v is not None else 0 for v in mlp_gen_improvements], width,
               label='MLP-Gen Δ', color='#9b59b6', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy Change vs Baseline (%)', fontsize=12)
    ax.set_title('Improvements Over Baseline', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels for improvements
    for i, (st, mc, mg) in enumerate(zip(steered_improvements, mlp_mc_improvements, mlp_gen_improvements)):
        if st is not None and abs(st) > 0.5:
            ax.text(i - width, st, f'{st:+.1f}', ha='center', va='bottom' if st > 0 else 'top', fontsize=8)
        if mc is not None and abs(mc) > 0.5:
            ax.text(i, mc, f'{mc:+.1f}', ha='center', va='bottom' if mc > 0 else 'top', fontsize=8)
        if mg is not None and abs(mg) > 0.5:
            ax.text(i + width, mg, f'{mg:+.1f}', ha='center', va='bottom' if mg > 0 else 'top', fontsize=8)

    plt.tight_layout()
    output_file = output_dir / "steering_effectiveness.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_heatmap(results: list, output_dir: Path):
    """Create heatmap of performance across models and variants."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance Heatmap Across Models', fontsize=16, fontweight='bold')

    # Prepare data
    model_names = [r['model_info']['full_name'] for r in results]
    variants = ['baseline', 'steered', 'mlp_gen']
    variant_labels = ['Baseline', 'Raw CAA', 'MLP-Gen']

    # Accuracy heatmap
    acc_matrix = []
    for r in results:
        row = []
        for variant in variants:
            if variant in r['variants']:
                row.append(r['variants'][variant]['accuracy'] * 100)
            else:
                row.append(np.nan)
        acc_matrix.append(row)

    ax = axes[0]
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=35, vmax=60)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(variant_labels)
    ax.set_yticklabels(model_names)
    ax.set_title('Judge Accuracy (%)', fontsize=13, fontweight='bold')

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(variants)):
            if not np.isnan(acc_matrix[i][j]):
                text = ax.text(j, i, f'{acc_matrix[i][j]:.1f}',
                             ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax)

    # Semantic heatmap
    sem_matrix = []
    for r in results:
        row = []
        for variant in variants:
            if variant in r['variants']:
                row.append(r['variants'][variant]['semantic_mean'])
            else:
                row.append(np.nan)
        sem_matrix.append(row)

    ax = axes[1]
    im = ax.imshow(sem_matrix, cmap='RdYlGn', aspect='auto', vmin=0.50, vmax=0.65)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(variant_labels)
    ax.set_yticklabels(model_names)
    ax.set_title('Semantic Similarity Score', fontsize=13, fontweight='bold')

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(variants)):
            if not np.isnan(sem_matrix[i][j]):
                text = ax.text(j, i, f'{sem_matrix[i][j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    output_file = output_dir / "performance_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def generate_comparison_report(results: list, output_dir: Path):
    """Generate a markdown report with key findings."""
    report = []
    report.append("# Cross-Model Comparison Report")
    report.append("=" * 80)
    report.append("")
    report.append(f"Total models analyzed: {len(results)}")
    report.append("")

    # Summary table - ALL VARIANTS
    report.append("## Performance Summary - All Variants")
    report.append("")
    report.append("### Judge Accuracy")
    report.append("| Model | Baseline | Steered | MLP-MC | MLP-Gen |")
    report.append("|-------|----------|---------|--------|---------|")

    for r in results:
        name = r['model_info']['full_name']
        baseline = f"{r['variants']['baseline']['accuracy']*100:5.1f}%" if 'baseline' in r['variants'] else "  -  "
        steered = f"{r['variants']['steered']['accuracy']*100:5.1f}%" if 'steered' in r['variants'] else "  -  "
        mlp_mc = f"{r['variants']['mlp_mc']['accuracy']*100:5.1f}%" if 'mlp_mc' in r['variants'] else "  -  "
        mlp_gen = f"{r['variants']['mlp_gen']['accuracy']*100:5.1f}%" if 'mlp_gen' in r['variants'] else "  -  "
        report.append(f"| {name:12} | {baseline:^8} | {steered:^7} | {mlp_mc:^6} | {mlp_gen:^7} |")

    report.append("")
    report.append("### Semantic Similarity")
    report.append("| Model | Baseline | Steered | MLP-MC | MLP-Gen |")
    report.append("|-------|----------|---------|--------|---------|")

    for r in results:
        name = r['model_info']['full_name']
        baseline = f"{r['variants']['baseline']['semantic_mean']:.4f}" if 'baseline' in r['variants'] else "  -   "
        steered = f"{r['variants']['steered']['semantic_mean']:.4f}" if 'steered' in r['variants'] else "  -   "
        mlp_mc = f"{r['variants']['mlp_mc']['semantic_mean']:.4f}" if 'mlp_mc' in r['variants'] else "  -   "
        mlp_gen = f"{r['variants']['mlp_gen']['semantic_mean']:.4f}" if 'mlp_gen' in r['variants'] else "  -   "
        report.append(f"| {name:12} | {baseline:^8} | {steered:^7} | {mlp_mc:^6} | {mlp_gen:^7} |")

    report.append("")
    report.append("### Improvements over Baseline")
    report.append("| Model | Steered Δ | MLP-MC Δ | MLP-Gen Δ |")
    report.append("|-------|-----------|----------|-----------|")

    for r in results:
        name = r['model_info']['full_name']
        if 'baseline' in r['variants']:
            b_acc = r['variants']['baseline']['accuracy'] * 100
            steered_delta = (r['variants']['steered']['accuracy'] * 100 - b_acc) if 'steered' in r['variants'] else None
            mlp_mc_delta = (r['variants']['mlp_mc']['accuracy'] * 100 - b_acc) if 'mlp_mc' in r['variants'] else None
            mlp_gen_delta = (r['variants']['mlp_gen']['accuracy'] * 100 - b_acc) if 'mlp_gen' in r['variants'] else None

            steered_str = f"{steered_delta:+5.1f}%" if steered_delta is not None else "  -  "
            mlp_mc_str = f"{mlp_mc_delta:+5.1f}%" if mlp_mc_delta is not None else "  -  "
            mlp_gen_str = f"{mlp_gen_delta:+5.1f}%" if mlp_gen_delta is not None else "  -  "
            report.append(f"| {name:12} | {steered_str:^9} | {mlp_mc_str:^8} | {mlp_gen_str:^9} |")

    report.append("")

    # Key findings
    report.append("## Key Findings")
    report.append("")

    # Best performing model
    best_baseline = max(results, key=lambda r: r['variants']['baseline']['accuracy'] if 'baseline' in r['variants'] else 0)
    report.append(f"### Best Baseline Performance")
    report.append(f"- **{best_baseline['model_info']['full_name']}**: "
                 f"{best_baseline['variants']['baseline']['accuracy']*100:.2f}% accuracy")
    report.append("")

    # Biggest improvement
    improvements = []
    for r in results:
        if 'baseline' in r['variants'] and 'mlp_gen' in r['variants']:
            delta = (r['variants']['mlp_gen']['accuracy'] - r['variants']['baseline']['accuracy']) * 100
            improvements.append((r, delta))

    if improvements:
        best_impr = max(improvements, key=lambda x: x[1])
        worst_impr = min(improvements, key=lambda x: x[1])

        report.append(f"### Biggest MLP-Gen Improvement")
        report.append(f"- **{best_impr[0]['model_info']['full_name']}**: {best_impr[1]:+.2f}%")
        report.append("")

        if worst_impr[1] < 0:
            report.append(f"### Biggest MLP-Gen Degradation")
            report.append(f"- **{worst_impr[0]['model_info']['full_name']}**: {worst_impr[1]:+.2f}%")
            report.append("")

    # Scaling trends
    report.append("### Scaling Observations")
    report.append("")

    gemma2_models = [r for r in results if r['model_info']['family'] == 'gemma2']
    if len(gemma2_models) >= 2:
        smallest = min(gemma2_models, key=lambda r: r['model_info']['size'])
        largest = max(gemma2_models, key=lambda r: r['model_info']['size'])

        if 'baseline' in smallest['variants'] and 'baseline' in largest['variants']:
            delta = (largest['variants']['baseline']['accuracy'] -
                    smallest['variants']['baseline']['accuracy']) * 100
            report.append(f"**Gemma2 Scaling**: Baseline accuracy improves by {delta:+.2f}% "
                         f"from {smallest['model_info']['size_str']} to {largest['model_info']['size_str']}")

    gemma3_models = [r for r in results if r['model_info']['family'] == 'gemma3']
    if len(gemma3_models) >= 2:
        smallest = min(gemma3_models, key=lambda r: r['model_info']['size'])
        largest = max(gemma3_models, key=lambda r: r['model_info']['size'])

        if 'baseline' in smallest['variants'] and 'baseline' in largest['variants']:
            delta = (largest['variants']['baseline']['accuracy'] -
                    smallest['variants']['baseline']['accuracy']) * 100
            report.append(f"**Gemma3 Scaling**: Baseline accuracy improves by {delta:+.2f}% "
                         f"from {smallest['model_info']['size_str']} to {largest['model_info']['size_str']}")

    report.append("")
    report.append("=" * 80)

    # Write report
    output_file = output_dir / "cross_model_comparison.md"
    with output_file.open('w') as f:
        f.write('\n'.join(report))

    print(f"✅ Saved: {output_file}")

    # Also print to console
    print("\n" + '\n'.join(report))


def main():
    base_dir = Path("outputs")
    output_dir = Path("outputs/cross_model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CROSS-MODEL COMPARISON ANALYSIS")
    print("=" * 80)
    print()

    print("📊 Loading results from all models...")
    results = load_all_results(base_dir)

    if not results:
        print("❌ No results found!")
        return

    print(f"✅ Loaded {len(results)} models:")
    for r in results:
        print(f"   - {r['model_info']['full_name']}")
    print()

    print("📈 Generating scaling analysis...")
    plot_scaling_analysis(results, output_dir)

    print("📊 Generating steering effectiveness comparison...")
    plot_steering_effectiveness(results, output_dir)

    print("🔥 Generating performance heatmaps...")
    plot_heatmap(results, output_dir)

    print("📝 Generating comparison report...")
    generate_comparison_report(results, output_dir)

    print()
    print("=" * 80)
    print(f"✅ All comparisons saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
