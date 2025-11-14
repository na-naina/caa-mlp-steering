#!/usr/bin/env python3
"""
Quick analysis of intermediate results (before job completes)
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def analyze_intermediate_results(results_dir):
    """Analyze intermediate JSON files"""

    results_path = Path(results_dir)

    # Find all intermediate files
    intermediate_files = list(results_path.glob("**/layer_*_scale_*_intermediate.json"))

    if not intermediate_files:
        print(f"No intermediate files found in {results_dir}")
        return

    print(f"Found {len(intermediate_files)} intermediate result files\n")

    # Extract data
    rows = []
    for file in sorted(intermediate_files):
        with open(file) as f:
            data = json.load(f)

        scale = data.get("scale", 0)

        # Get the result key (baseline, caa_scale_X, etc.)
        result_keys = [k for k in data.keys() if k not in ['scale', 'timestamp']]

        for key in result_keys:
            result = data[key]

            # Extract MC results
            mc = result.get("mc_binary", {})
            mc_acc = mc.get("accuracy", 0)
            mc_correct_prob = mc.get("avg_correct_prob", 0)
            mc_incorrect_prob = mc.get("avg_incorrect_prob", 0)

            # Extract open-ended results
            oe = result.get("open_ended", {})
            oe_truth = oe.get("truthfulness", 0)
            oe_info = oe.get("informativeness", 0)

            rows.append({
                "file": file.parent.name,
                "condition": key,
                "scale": scale,
                "mc_accuracy": f"{mc_acc:.3f}",
                "mc_correct_prob": f"{mc_correct_prob:.3f}",
                "mc_incorrect_prob": f"{mc_incorrect_prob:.3f}",
                "oe_truthfulness": f"{oe_truth:.3f}",
                "oe_informativeness": f"{oe_info:.3f}",
                "mc_samples": mc.get("total_samples", 0),
                "oe_samples": oe.get("total_samples", 0)
            })

    df = pd.DataFrame(rows)

    # Print summary
    print("="*80)
    print("INTERMEDIATE RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print()

    # Group by model and show scaling effect
    print("\n" + "="*80)
    print("SCALING EFFECTS (MC Accuracy)")
    print("="*80)

    for model in df['file'].unique():
        model_df = df[df['file'] == model].copy()
        model_df['scale'] = pd.to_numeric(model_df['scale'])
        model_df['mc_accuracy'] = pd.to_numeric(model_df['mc_accuracy'])
        model_df = model_df.sort_values('scale')

        print(f"\n{model}:")
        print(f"  Scale    MC Acc    Change from baseline")
        print(f"  -----    ------    -------------------")

        baseline = model_df[model_df['scale'] == 0]['mc_accuracy'].values[0] if 0 in model_df['scale'].values else 0

        for _, row in model_df.iterrows():
            scale = row['scale']
            acc = row['mc_accuracy']
            change = acc - baseline if baseline > 0 else 0
            sign = "+" if change > 0 else ""
            print(f"  {scale:5.1f}    {acc:.3f}    {sign}{change:.3f}")

    # Save to CSV
    output_file = results_path / "intermediate_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nSaved CSV to: {output_file}")

    return df, intermediate_files

def create_visualizations(df, results_path, intermediate_files):
    """Create visualizations of results"""

    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Create output directory for plots
    plot_dir = results_path / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Extract judge model from intermediate files (if available)
    judge_model = "Unknown"
    if intermediate_files:
        try:
            with open(intermediate_files[0]) as f:
                data = json.load(f)
                if 'judge_model' in data:
                    judge_model = data['judge_model']
        except:
            pass

    # Plot for each model: separate MC and Open-ended
    for model in df['file'].unique():
        model_df = df[df['file'] == model].copy()
        model_df['scale'] = pd.to_numeric(model_df['scale'])
        model_df['mc_accuracy'] = pd.to_numeric(model_df['mc_accuracy'])
        model_df['oe_truthfulness'] = pd.to_numeric(model_df['oe_truthfulness'])
        model_df = model_df.sort_values('scale')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: MC Accuracy with sample counts as labels
        ax1.plot(model_df['scale'], model_df['mc_accuracy'], 'o-', linewidth=2, markersize=8, color='steelblue')
        baseline = model_df[model_df['scale'] == 0]['mc_accuracy'].values[0] if 0 in model_df['scale'].values else None
        if baseline is not None:
            ax1.axhline(y=baseline, color='red', linestyle='--', alpha=0.7, label='Baseline')

        # Add sample counts as labels on points
        for _, row in model_df.iterrows():
            ax1.annotate(f"{int(row['mc_samples'])} samples",
                        xy=(row['scale'], row['mc_accuracy']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

        ax1.set_xlabel('CAA Steering Scale', fontsize=12)
        ax1.set_ylabel('MC Accuracy', fontsize=12)
        ax1.set_title(f'{model} - Multiple Choice Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Open-ended Truthfulness with judge model label
        ax2.plot(model_df['scale'], model_df['oe_truthfulness'], 'o-', linewidth=2, markersize=8, color='forestgreen')
        baseline_oe = model_df[model_df['scale'] == 0]['oe_truthfulness'].values[0] if 0 in model_df['scale'].values else None
        if baseline_oe is not None:
            ax2.axhline(y=baseline_oe, color='red', linestyle='--', alpha=0.7, label='Baseline')

        ax2.set_xlabel('CAA Steering Scale', fontsize=12)
        ax2.set_ylabel('Truthfulness Score', fontsize=12)
        ax2.set_title(f'{model} - Open-ended Truthfulness\nJudge: {judge_model}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plot_file = plot_dir / f"{model}_scaling.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_file.name}")

    # Comparison plots across all models
    if len(df['file'].unique()) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # MC Comparison
        for model in df['file'].unique():
            model_df = df[df['file'] == model].copy()
            model_df['scale'] = pd.to_numeric(model_df['scale'])
            model_df['mc_accuracy'] = pd.to_numeric(model_df['mc_accuracy'])
            model_df = model_df.sort_values('scale')
            ax1.plot(model_df['scale'], model_df['mc_accuracy'], 'o-', linewidth=2,
                   markersize=8, label=model, alpha=0.8)

        ax1.set_xlabel('CAA Steering Scale', fontsize=12)
        ax1.set_ylabel('MC Accuracy', fontsize=12)
        ax1.set_title('Multiple Choice Accuracy - All Models', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Open-ended Comparison
        for model in df['file'].unique():
            model_df = df[df['file'] == model].copy()
            model_df['scale'] = pd.to_numeric(model_df['scale'])
            model_df['oe_truthfulness'] = pd.to_numeric(model_df['oe_truthfulness'])
            model_df = model_df.sort_values('scale')
            ax2.plot(model_df['scale'], model_df['oe_truthfulness'], 'o-', linewidth=2,
                   markersize=8, label=model, alpha=0.8)

        ax2.set_xlabel('CAA Steering Scale', fontsize=12)
        ax2.set_ylabel('Truthfulness Score', fontsize=12)
        ax2.set_title(f'Open-ended Truthfulness - All Models\nJudge: {judge_model}', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = plot_dir / "all_models_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_file.name}")

    print(f"\nAll plots saved to: {plot_dir}/")

def show_sample_outputs(intermediate_files, num_samples=2):
    """Show example outputs for both MC and open-ended at different scales"""

    print("\n" + "="*80)
    print("SAMPLE OUTPUTS AT DIFFERENT SCALES")
    print("="*80)

    # Group files by scale
    files_by_scale = {}
    for file in intermediate_files:
        with open(file) as f:
            data = json.load(f)
        scale = data.get('scale', 'unknown')
        if scale not in files_by_scale:
            files_by_scale[scale] = []
        files_by_scale[scale].append((file, data))

    # Show examples for a few representative scales
    scales_to_show = sorted([s for s in files_by_scale.keys() if s != 'unknown'])[:3]  # First 3 scales

    for scale in scales_to_show:
        if not files_by_scale[scale]:
            continue

        file, data = files_by_scale[scale][0]
        print(f"\n{'='*80}")
        print(f"SCALE: {scale}")
        print(f"File: {file.name}")
        print(f"{'='*80}")

        # Get the result key
        result_keys = [k for k in data.keys() if k not in ['scale', 'timestamp', 'judge_model']]
        if not result_keys:
            continue

        result = data[result_keys[0]]

        # Show MC examples
        mc_results = result.get("mc_binary", {})
        mc_samples = mc_results.get("sample_results", [])

        if mc_samples:
            print(f"\n--- MULTIPLE CHOICE EXAMPLES ---")
            for i, sample in enumerate(mc_samples[:num_samples], 1):
                print(f"\nMC Example {i}:")
                print(f"Question: {sample.get('question', 'N/A')}")
                print(f"A) {sample.get('choice_a', 'N/A')}")
                print(f"B) {sample.get('choice_b', 'N/A')}")
                print(f"Correct: {sample.get('correct_label', 'N/A')}")
                print(f"Predicted: {sample.get('predicted', 'N/A')}")
                print(f"Generated token: {sample.get('generated', 'N/A')}")
                correct = "✓" if sample.get('predicted') == sample.get('correct_label') else "✗"
                print(f"Result: {correct}")

        # Show Open-ended examples
        oe_results = result.get("open_ended", {})
        oe_samples = oe_results.get("sample_results", [])

        if oe_samples:
            print(f"\n--- OPEN-ENDED EXAMPLES ---")
            print("\nExplanation: Open-ended evaluation tests the model's ability to generate")
            print("truthful free-form answers. A judge model evaluates each generated answer.")

            for i, sample in enumerate(oe_samples[:num_samples], 1):
                print(f"\n{'-'*80}")
                print(f"Open-ended Example {i}:")
                print(f"{'-'*80}")

                # Original question and what model generated
                print(f"\n📝 Question Asked to Model:")
                print(f"   {sample.get('question', 'N/A')}")

                print(f"\n🤖 Model's Generated Answer (with CAA steering):")
                print(f"   {sample.get('generated', 'N/A')}")

                # Reference answers
                print(f"\n✓ Known Correct Answer(s):")
                true_answers = sample.get('true_answers', [])
                if isinstance(true_answers, list):
                    for ans in true_answers[:2]:  # Show first 2
                        print(f"   - {ans}")
                else:
                    print(f"   {sample.get('best_answer', 'N/A')}")

                print(f"\n✗ Known Incorrect Answer(s):")
                false_answers = sample.get('false_answers', [])
                if isinstance(false_answers, list):
                    for ans in false_answers[:2]:  # Show first 2
                        print(f"   - {ans}")

                # Judge evaluation
                print(f"\n⚖️  Judge Evaluation:")
                if 'judge_prompt' in sample:
                    print(f"\n   Judge was given this prompt:")
                    print(f"   {'-'*60}")
                    print(f"   {sample['judge_prompt'][:400]}...")  # First 400 chars
                    print(f"   {'-'*60}")

                if 'judge_response' in sample:
                    print(f"\n   Judge's Full Response:")
                    print(f"   >>> {sample['judge_response']}")

                # Parsed results
                if 'truthful' in sample:
                    truthful = "✓ YES" if sample['truthful'] == 1 else "✗ NO"
                    print(f"\n   Parsed - Truthful: {truthful}")
                if 'informative' in sample:
                    informative = "✓ YES" if sample['informative'] == 1 else "✗ NO"
                    print(f"   Parsed - Informative: {informative}")

def analyze_intermediate_results(results_dir, create_plots=True, show_samples=True):
    """Analyze intermediate JSON files"""

    results_path = Path(results_dir)

    # Find all intermediate files
    intermediate_files = list(results_path.glob("**/layer_*_scale_*_intermediate.json"))

    if not intermediate_files:
        print(f"No intermediate files found in {results_dir}")
        return

    print(f"Found {len(intermediate_files)} intermediate result files\n")

    # Extract data
    rows = []
    for file in sorted(intermediate_files):
        with open(file) as f:
            data = json.load(f)

        scale = data.get("scale", 0)

        # Get the result key (baseline, caa_scale_X, etc.)
        result_keys = [k for k in data.keys() if k not in ['scale', 'timestamp']]

        for key in result_keys:
            result = data[key]

            # Extract MC results
            mc = result.get("mc_binary", {})
            mc_acc = mc.get("accuracy", 0)
            mc_correct_prob = mc.get("avg_correct_prob", 0)
            mc_incorrect_prob = mc.get("avg_incorrect_prob", 0)

            # Extract open-ended results
            oe = result.get("open_ended", {})
            oe_truth = oe.get("truthfulness", 0)
            oe_info = oe.get("informativeness", 0)

            rows.append({
                "file": file.parent.name,
                "condition": key,
                "scale": scale,
                "mc_accuracy": f"{mc_acc:.3f}",
                "mc_correct_prob": f"{mc_correct_prob:.3f}",
                "mc_incorrect_prob": f"{mc_incorrect_prob:.3f}",
                "oe_truthfulness": f"{oe_truth:.3f}",
                "oe_informativeness": f"{oe_info:.3f}",
                "mc_samples": mc.get("total_samples", 0),
                "oe_samples": oe.get("total_samples", 0)
            })

    df = pd.DataFrame(rows)

    # Print summary
    print("="*80)
    print("INTERMEDIATE RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print()

    # Group by model and show scaling effect
    print("\n" + "="*80)
    print("SCALING EFFECTS (MC Accuracy)")
    print("="*80)

    for model in df['file'].unique():
        model_df = df[df['file'] == model].copy()
        model_df['scale'] = pd.to_numeric(model_df['scale'])
        model_df['mc_accuracy'] = pd.to_numeric(model_df['mc_accuracy'])
        model_df = model_df.sort_values('scale')

        print(f"\n{model}:")
        print(f"  Scale    MC Acc    Change from baseline")
        print(f"  -----    ------    -------------------")

        baseline = model_df[model_df['scale'] == 0]['mc_accuracy'].values[0] if 0 in model_df['scale'].values else 0

        for _, row in model_df.iterrows():
            scale = row['scale']
            acc = row['mc_accuracy']
            change = acc - baseline if baseline > 0 else 0
            sign = "+" if change > 0 else ""
            print(f"  {scale:5.1f}    {acc:.3f}    {sign}{change:.3f}")

    # Save to CSV
    output_file = results_path / "intermediate_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nSaved CSV to: {output_file}")

    # Create visualizations
    if create_plots:
        create_visualizations(df, results_path, intermediate_files)

    # Show sample outputs
    if show_samples:
        show_sample_outputs(intermediate_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze intermediate CAA results")
    parser.add_argument("--results_dir", type=str,
                       default="/springbrook/share/dcsresearch/u5584851/caa_experiments/results",
                       help="Directory containing results")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip creating plots (faster)")
    parser.add_argument("--no-samples", action="store_true",
                       help="Skip showing sample outputs")

    args = parser.parse_args()
    analyze_intermediate_results(args.results_dir,
                                 create_plots=not args.no_plots,
                                 show_samples=not args.no_samples)
