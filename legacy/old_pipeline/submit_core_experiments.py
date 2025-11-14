#!/usr/bin/env python3
"""
Submit core CAA experiments for Gemma2 and Gemma3 models to SLURM
Tests: Gemma2-2B, Gemma2-9B, Gemma2-27B, Gemma3-1B, Gemma3-9B, Gemma3-27B
"""

import subprocess
import time
from pathlib import Path
import sys

# Core model configurations
CORE_EXPERIMENTS = [
    # Model Key, Model Name, Optimal Layer, Use MLP, Scale Set, Eval Set

    # === GEMMA 2 FAMILY ===
    ("gemma2-2b", "google/gemma-2-2b", 12, False, "standard", "standard"),
    ("gemma2-2b-mlp", "google/gemma-2-2b", 12, True, "standard", "standard"),

    ("gemma2-9b", "google/gemma-2-9b", 21, False, "standard", "standard"),
    ("gemma2-9b-mlp", "google/gemma-2-9b", 21, True, "standard", "standard"),

    ("gemma2-27b", "google/gemma-2-27b", 24, False, "coarse", "quick"),
    ("gemma2-27b-mlp", "google/gemma-2-27b", 24, True, "coarse", "quick"),

    # === GEMMA 3 FAMILY ===
    # Note: Gemma3 models don't support top_p/top_k generation parameters
    # The code automatically detects Gemma3 and uses temperature-only sampling
    ("gemma3-1b", "google/gemma-3-1b-pt", 9, False, "standard", "standard"),
    ("gemma3-1b-mlp", "google/gemma-3-1b-pt", 9, True, "standard", "standard"),

    # Larger Gemma3 models (multimodal but work in text-only mode)
    ("gemma3-12b", "google/gemma-3-12b-pt", 21, False, "standard", "standard"),
    ("gemma3-12b-mlp", "google/gemma-3-12b-pt", 21, True, "standard", "standard"),

    ("gemma3-27b", "google/gemma-3-27b-pt", 26, False, "coarse", "quick"),
    ("gemma3-27b-mlp", "google/gemma-3-27b-pt", 26, True, "coarse", "quick"),
]

def get_slurm_resources(model_key):
    """Get appropriate SLURM resources based on model size"""
    if "1b" in model_key or "2b" in model_key:
        return {
            "cpus": 10,  # Max 10:1 CPU:GPU ratio per Blythe documentation
            "mem": "64000",  # 64GB in MB
            "time": "06:00:00",
            "gres": "gpu:lovelace_l40:1",  # Specific GPU type for Blythe
            "partition": "gpu",  # GPU partition
        }
    elif "9b" in model_key:
        return {
            "cpus": 10,  # Max 10:1 CPU:GPU ratio per Blythe documentation
            "mem": "128000",  # 128GB in MB
            "time": "12:00:00",
            "gres": "gpu:lovelace_l40:1",
            "partition": "gpu",
        }
    elif "12b" in model_key:
        return {
            "cpus": 10,  # Max 10:1 CPU:GPU ratio per Blythe documentation
            "mem": "160000",  # 160GB in MB
            "time": "18:00:00",
            "gres": "gpu:lovelace_l40:1",
            "partition": "gpu",
        }
    elif "27b" in model_key:
        return {
            "cpus": 20,  # 10:1 ratio for 2 GPUs (10 CPUs per GPU)
            "mem": "180000",  # 180GB in MB (leaving some headroom)
            "time": "24:00:00",
            "gres": "gpu:lovelace_l40:2",  # 2 L40 GPUs for large models
            "partition": "gpu",
        }
    else:
        return {
            "cpus": 10,  # Max 10:1 CPU:GPU ratio per Blythe documentation
            "mem": "64000",
            "time": "12:00:00",
            "gres": "gpu:lovelace_l40:1",
            "partition": "gpu",
        }

def create_job_script(exp_config, job_id):
    """Create a temporary job script for the experiment"""

    model_key, model_name, layer, use_mlp, scale_set, eval_set = exp_config
    resources = get_slurm_resources(model_key)

    # Determine judge model
    if "27b" in model_key:
        judge_model = "google/gemma-3-9b-it"  # Use smaller judge for 27B models
    else:
        judge_model = "google/gemma-3-27b-it"

    # Scale configurations
    scale_map = {
        "coarse": "0 1.0 5.0 10.0",
        "standard": "0 0.5 1.0 2.0 5.0 10.0",
        "fine": "0 0.1 0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.5 10.0",
    }

    # Evaluation configurations
    eval_map = {
        "quick": (50, 50),
        "standard": (200, 200),
        "full": (817, 817),
    }

    scales = scale_map.get(scale_set, "0 1.0 5.0")
    num_mc, num_gen = eval_map.get(eval_set, (100, 100))

    script_content = f"""#!/bin/bash
#SBATCH --job-name=caa_{model_key}_L{layer}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={resources['cpus']}
#SBATCH --mem={resources['mem']}
#SBATCH --time={resources['time']}
#SBATCH --gres={resources['gres']}
#SBATCH --partition={resources['partition']}
#SBATCH --output=logs/{model_key}_L{layer}_%j.out
#SBATCH --error=logs/{model_key}_L{layer}_%j.err

# Setup environment
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

# Activate virtual environment
source venv/bin/activate

# IMPORTANT: Do NOT set CUDA_VISIBLE_DEVICES manually
# SLURM sets this automatically based on the GPU allocation
# Setting it manually will cause conflicts and hanging (per Blythe documentation)

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# CRITICAL: Run Python in unbuffered mode to see real-time output
export PYTHONUNBUFFERED=1

# Setup shared storage environment
SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
PROJECT_DIR="$SHARED_DIR/caa_experiments"

# Check if shared directory is accessible
if [ -d "$SHARED_DIR" ]; then
    echo "Using shared storage at $SHARED_DIR"

    # Create directories in shared space
    mkdir -p "$PROJECT_DIR/cache/huggingface"
    mkdir -p "$PROJECT_DIR/cache/transformers"
    mkdir -p "$PROJECT_DIR/results"

    # Set environment to use shared cache
    export HF_HOME="$PROJECT_DIR/cache/huggingface"
    export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/transformers"
    export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"
    export HUGGINGFACE_HUB_CACHE="$PROJECT_DIR/cache/huggingface/hub"

    # Copy token if needed
    if [ -f ~/.cache/huggingface/token ] && [ ! -f "$HF_HOME/token" ]; then
        cp ~/.cache/huggingface/token "$HF_HOME/token"
    fi

    # Load token
    if [ -f "$HF_HOME/token" ]; then
        export HF_TOKEN=$(cat "$HF_HOME/token")
        export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    fi

    OUTPUT_BASE="$PROJECT_DIR/results"
else
    echo "WARNING: Shared directory not accessible, using local cache"
    export HF_HOME=$PWD/.cache/huggingface
    export TRANSFORMERS_CACHE=$PWD/.cache/transformers
    export HUGGINGFACE_HUB_CACHE=$PWD/.cache/huggingface/hub

    mkdir -p $HF_HOME
    mkdir -p $TRANSFORMERS_CACHE

    if [ -f ~/.cache/huggingface/token ]; then
        export HF_TOKEN=$(cat ~/.cache/huggingface/token)
        export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
        cp ~/.cache/huggingface/token $HF_HOME/token
    fi

    OUTPUT_BASE="$PWD/results"
fi

# Run experiment
python caa_truthfulqa.py \\
    --model_name {model_name} \\
    --layer {layer} \\
    --scales {scales} \\
    --caa_samples 100 \\
    --num_mc_samples {num_mc} \\
    --num_gen_samples {num_gen} \\
    --judge_model {judge_model} \\
    --output_dir $OUTPUT_BASE/{model_key} \\
    {"--use_mlp" if use_mlp else ""}

echo "Experiment completed for {model_key} at layer {layer}"
"""

    script_path = f"temp_job_{job_id}.slurm"
    with open(script_path, "w") as f:
        f.write(script_content)

    return script_path

def submit_job(exp_config, dry_run=False):
    """Submit a single experiment to SLURM"""

    model_key = exp_config[0]
    job_id = f"{model_key}_{int(time.time())}"

    # Create temporary job script
    script_path = create_job_script(exp_config, job_id)

    if dry_run:
        print(f"[DRY RUN] Would submit: {model_key}")
        with open(script_path, "r") as f:
            print(f.read())
        Path(script_path).unlink()  # Clean up
        return None

    try:
        # Submit job
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True
        )

        # Extract job ID
        slurm_job_id = result.stdout.strip().split()[-1]
        print(f"✓ Submitted {model_key}: Job ID {slurm_job_id}")

        # Clean up temporary script
        Path(script_path).unlink()

        return slurm_job_id

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit {model_key}: {e.stderr}")
        Path(script_path).unlink()  # Clean up even on failure
        return None

def main():
    """Main submission function"""

    import argparse
    parser = argparse.ArgumentParser(description="Submit core CAA experiments")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between submissions (seconds)")
    parser.add_argument("--filter", type=str, default=None,
                       help="Filter experiments by model key pattern")
    args = parser.parse_args()

    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    print("=" * 60)
    print("CAA Core Experiments Submission")
    print("=" * 60)
    print()

    # Filter experiments if requested
    experiments = CORE_EXPERIMENTS
    if args.filter:
        experiments = [e for e in experiments if args.filter in e[0]]
        print(f"Filtered to {len(experiments)} experiments matching '{args.filter}'")

    print(f"Total experiments to submit: {len(experiments)}")
    print()

    # List experiments
    print("Experiments:")
    for i, exp in enumerate(experiments, 1):
        model_key, model_name, layer, use_mlp, _, _ = exp
        mlp_str = " (+MLP)" if use_mlp else ""
        print(f"  {i}. {model_key} (Layer {layer}){mlp_str}")
    print()

    if args.dry_run:
        print("DRY RUN MODE - No jobs will be submitted")
        print()

    # Submit experiments
    job_ids = []
    for exp in experiments:
        job_id = submit_job(exp, dry_run=args.dry_run)
        if job_id:
            job_ids.append(job_id)
        time.sleep(args.delay)  # Small delay between submissions

    # Summary
    print()
    print("=" * 60)
    if not args.dry_run:
        print(f"Successfully submitted {len(job_ids)}/{len(experiments)} jobs")

        if job_ids:
            print("\nTo check status:")
            print(f"  squeue -j {','.join(job_ids)}")
            print("\nTo cancel all:")
            print(f"  scancel {' '.join(job_ids)}")
    else:
        print("Dry run completed")

    return 0 if len(job_ids) == len(experiments) else 1

if __name__ == "__main__":
    sys.exit(main())