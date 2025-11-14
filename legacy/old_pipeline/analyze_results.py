"""
Analysis script for CAA experiment results
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import argparse

def load_experiment_results(results_dir: Path) -> Dict:
    """Load all results from a directory"""
    
    all_results = {}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        all_results[model_name] = {}
        
        # Load all layer results
        for result_file in model_dir.glob("layer_*_results.json"):
            layer_num = int(result_file.stem.split("_")[1])
            
            with open(result_file, "r") as f:
                data = json.load(f)
                
            all_results[model_name][layer_num] = data
    
    return all_results

def extract_metrics_table(results: Dict) -> pd.DataFrame:
    """Extract metrics into a pandas DataFrame"""
    
    rows = []
    
    for model_name, model_results in results.items():
        for layer, layer_results in model_results.items():
            for condition, cond_results in layer_results.items():
                
                if "metrics" not in cond_results:
                    continue
                    
                # Parse condition name
                if condition == "baseline":
                    scale = 0
                    method = "baseline"
                elif condition.startswith("caa_mcp"):
                    scale = float(condition.split("_")[-1])
                    method = "caa_mcp"
                elif condition.startswith("caa"):
                    scale = float(condition.split("_")[-1])
                    method = "caa"
                else:
                    continue
                
                rows.append({
                    "model": model_name,
                    "layer": layer,
                    "method": method,
                    "scale": scale,
                    "accuracy": cond_results["metrics"]["accuracy"],
                    "num_samples": cond_results["metrics"]["total_samples"]
                })
    
    return pd.DataFrame(rows)

def plot_scaling_curves(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy vs scale curves for each model and method"""
    
    models = df["model"].unique()
    
    for model in models:
        model_df = df[df["model"] == model]
        
        # Get optimal layer (most common in results)
        optimal_layer = model_df["layer"].value_counts().index[0]
        layer_df = model_df[model_df["layer"] == optimal_layer]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: CAA vs baseline
        for method in ["baseline", "caa", "caa_mcp"]:
            method_df = layer_df[layer_df["method"] == method]
            if not method_df.empty:
                method_df = method_df.sort_values("scale")
                
                if method == "baseline":
                    ax1.axhline(y=method_df["accuracy"].iloc[0], 
                              color='gray', linestyle='--', label='Baseline')
                else:
                    ax1.plot(method_df["scale"], method_df["accuracy"], 
                           marker='o', label=method.upper())
        
        ax1.set_xlabel("Steering Scale")
        ax1.set_ylabel("TruthfulQA Accuracy")
        ax1.set_title(f"{model} - Layer {optimal_layer}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Layer comparison (if available)
        layers = model_df["layer"].unique()
        if len(layers) > 1:
            for layer in sorted(layers):
                layer_data = model_df[(model_df["layer"] == layer) & 
                                     (model_df["method"] == "caa")]
                if not layer_data.empty:
                    layer_data = layer_data.sort_values("scale")
                    ax2.plot(layer_data["scale"], layer_data["accuracy"],
                           marker='o', label=f"Layer {layer}")
            
            ax2.set_xlabel("Steering Scale")
            ax2.set_ylabel("TruthfulQA Accuracy")
            ax2.set_title(f"{model} - Layer Comparison")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No layer comparison data", 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        # Save figure
        output_file = output_dir / f"{model.replace('/', '_')}_scaling_curves.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {output_file}")

def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table of best results"""
    
    summary_rows = []
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        
        # Baseline accuracy
        baseline = model_df[model_df["method"] == "baseline"]["accuracy"].mean()
        
        # Best CAA result
        caa_df = model_df[model_df["method"] == "caa"]
        if not caa_df.empty:
            best_caa_idx = caa_df["accuracy"].idxmax()
            best_caa = caa_df.loc[best_caa_idx]
            caa_improvement = best_caa["accuracy"] - baseline
        else:
            best_caa = None
            caa_improvement = 0
        
        # Best CAA+MCP result
        mcp_df = model_df[model_df["method"] == "caa_mcp"]
        if not mcp_df.empty:
            best_mcp_idx = mcp_df["accuracy"].idxmax()
            best_mcp = mcp_df.loc[best_mcp_idx]
            mcp_improvement = best_mcp["accuracy"] - baseline
        else:
            best_mcp = None
            mcp_improvement = 0
        
        summary_rows.append({
            "Model": model,
            "Baseline Acc": f"{baseline:.3f}",
            "Best CAA Acc": f"{best_caa['accuracy']:.3f}" if best_caa is not None else "N/A",
            "CAA Scale": f"{best_caa['scale']:.1f}" if best_caa is not None else "N/A",
            "CAA Improvement": f"{caa_improvement:+.3f}" if best_caa is not None else "N/A",
            "Best MCP Acc": f"{best_mcp['accuracy']:.3f}" if best_mcp is not None else "N/A",
            "MCP Scale": f"{best_mcp['scale']:.1f}" if best_mcp is not None else "N/A",
            "MCP Improvement": f"{mcp_improvement:+.3f}" if best_mcp is not None else "N/A"
        })
    
    return pd.DataFrame(summary_rows)

def analyze_layer_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze the effect of different layers"""
    
    layer_summary = []
    
    for model in df["model"].unique():
        model_df = df[(df["model"] == model) & (df["method"] == "caa")]
        
        if model_df.empty:
            continue
            
        # Group by layer and find best scale for each
        for layer in model_df["layer"].unique():
            layer_df = model_df[model_df["layer"] == layer]
            
            if not layer_df.empty:
                best_idx = layer_df["accuracy"].idxmax()
                best = layer_df.loc[best_idx]
                
                layer_summary.append({
                    "Model": model,
                    "Layer": layer,
                    "Best Accuracy": best["accuracy"],
                    "Optimal Scale": best["scale"]
                })
    
    layer_df = pd.DataFrame(layer_summary)
    
    if not layer_df.empty:
        # Sort by model and accuracy
        layer_df = layer_df.sort_values(["Model", "Best Accuracy"], ascending=[True, False])
    
    return layer_df

def generate_report(results_dir: Path, output_dir: Path):
    """Generate comprehensive analysis report"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading results...")
    results = load_experiment_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Extract metrics
    df = extract_metrics_table(results)
    
    # Save raw data
    df.to_csv(output_dir / "metrics_raw.csv", index=False)
    print(f"Saved raw metrics: {output_dir / 'metrics_raw.csv'}")
    
    # Create plots
    print("Creating plots...")
    plot_scaling_curves(df, output_dir)
    
    # Create summary tables
    print("Creating summary tables...")
    
    summary_df = create_summary_table(df)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    layer_df = analyze_layer_effects(df)
    if not layer_df.empty:
        layer_df.to_csv(output_dir / "layer_analysis.csv", index=False)
        print("\nLayer Analysis:")
        print(layer_df.to_string(index=False))
    
    # Create markdown report
    report = f"""# CAA Experiment Results

## Summary

Total models tested: {len(results)}
Total experiments: {len(df)}

## Best Results

{summary_df.to_markdown(index=False)}

## Layer Analysis

{layer_df.to_markdown(index=False) if not layer_df.empty else "No layer analysis available"}

## Plots

See the following files for detailed visualizations:
"""
    
    for plot_file in output_dir.glob("*.png"):
        report += f"\n- {plot_file.name}"
    
    with open(output_dir / "report.md", "w") as f:
        f.write(report)
    
    print(f"\nFull report saved to: {output_dir / 'report.md'}")

def main():
    parser = argparse.ArgumentParser(description="Analyze CAA experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                      help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="analysis",
                      help="Output directory for analysis")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    generate_report(results_dir, output_dir)

if __name__ == "__main__":
    main()
