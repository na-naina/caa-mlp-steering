"""
Batch runner for CAA experiments across multiple models
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
from typing import List, Dict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_configs import MODEL_CONFIGS, SCALE_CONFIGS, TRUTHFULQA_CONFIGS, PROMPT_SETS
from caa_experiment import run_experiment

def create_experiment_matrix(
    models: List[str],
    scale_granularity: str = "standard",
    eval_mode: str = "standard",
    use_mcp: bool = True
) -> List[Dict]:
    """Create matrix of experiments to run"""
    
    experiments = []
    scales = SCALE_CONFIGS[scale_granularity]
    eval_config = TRUTHFULQA_CONFIGS[eval_mode]
    
    for model_key in models:
        if model_key not in MODEL_CONFIGS:
            print(f"Warning: {model_key} not in configs, skipping...")
            continue
            
        config = MODEL_CONFIGS[model_key]
        
        # Test optimal layer with all scales
        experiments.append({
            "model_name": config["model_name"],
            "model_key": model_key,
            "layer": config["optimal_layer"],
            "scales": scales,
            "num_samples": eval_config["num_samples"],
            "use_mcp": use_mcp,
            "experiment_type": "optimal_layer"
        })
        
        # Test layer sweep with fewer scales
        if scale_granularity != "fine":  # Don't do layer sweep with fine scales
            for layer in config["layers"]:
                if layer != config["optimal_layer"]:  # Skip optimal (already done)
                    experiments.append({
                        "model_name": config["model_name"],
                        "model_key": model_key,
                        "layer": layer,
                        "scales": SCALE_CONFIGS["coarse"],  # Use coarse for layer sweep
                        "num_samples": TRUTHFULQA_CONFIGS["quick"]["num_samples"],
                        "use_mcp": use_mcp,
                        "experiment_type": "layer_sweep"
                    })
    
    return experiments

def run_single_experiment(exp_config: Dict, output_base: Path):
    """Run a single experiment configuration"""
    
    # Create command
    cmd = [
        "python", "caa_experiment.py",
        "--model_name", exp_config["model_name"],
        "--layer", str(exp_config["layer"]),
        "--scales"] + [str(s) for s in exp_config["scales"]] + [
        "--num_samples", str(exp_config["num_samples"]),
        "--output_dir", str(output_base)
    ]
    
    if exp_config["use_mcp"]:
        cmd.append("--use_mcp")
    
    # Add experiment metadata
    exp_dir = output_base / exp_config["model_key"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": exp_config
    }
    
    with open(exp_dir / f"experiment_metadata_layer_{exp_config['layer']}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Run experiment
    print(f"\nRunning: {exp_config['model_key']} - Layer {exp_config['layer']}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch runner for CAA experiments")
    parser.add_argument("--models", nargs="+", default=["gemma2-2b", "gemma2-9b"],
                      help="Model keys to test")
    parser.add_argument("--scale_granularity", choices=["fine", "standard", "coarse"],
                      default="standard", help="Granularity of scale sweep")
    parser.add_argument("--eval_mode", choices=["quick", "standard", "full"],
                      default="standard", help="Evaluation mode")
    parser.add_argument("--use_mcp", action="store_true", help="Use MCP layer")
    parser.add_argument("--output_dir", type=str, default="results",
                      help="Base output directory")
    parser.add_argument("--dry_run", action="store_true",
                      help="Print experiments without running")
    
    args = parser.parse_args()
    
    # Create experiment matrix
    experiments = create_experiment_matrix(
        models=args.models,
        scale_granularity=args.scale_granularity,
        eval_mode=args.eval_mode,
        use_mcp=args.use_mcp
    )
    
    print(f"Created {len(experiments)} experiment configurations")
    
    if args.dry_run:
        print("\nDry run - would run:")
        for exp in experiments:
            print(f"  {exp['model_key']} - Layer {exp['layer']} - {exp['experiment_type']}")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / f"batch_run_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    with open(output_base / "batch_config.json", "w") as f:
        json.dump({
            "args": vars(args),
            "experiments": experiments,
            "timestamp": timestamp
        }, f, indent=2)
    
    # Run experiments
    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"{'='*60}")
        
        success = run_single_experiment(exp, output_base)
        results.append({
            "experiment": exp,
            "success": success
        })
        
        # Save intermediate results
        with open(output_base / "batch_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH RUN COMPLETE")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r["success"])
    print(f"Successful: {successful}/{len(experiments)}")
    print(f"Results saved to: {output_base}")

if __name__ == "__main__":
    main()
