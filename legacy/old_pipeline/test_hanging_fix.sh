#!/bin/bash
#SBATCH --job-name=test_hang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64000
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --partition=gpu
#SBATCH --output=logs/test_hang_%j.out
#SBATCH --error=logs/test_hang_%j.err

echo "=================================================="
echo "Testing Script Exit Behavior"
echo "=================================================="
echo ""

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

source venv/bin/activate

# Critical: unbuffered output
export PYTHONUNBUFFERED=1

# Shared storage setup
SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
PROJECT_DIR="$SHARED_DIR/caa_experiments"

if [ -d "$SHARED_DIR" ]; then
    export HF_HOME="$PROJECT_DIR/cache/huggingface"
    export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/transformers"
    export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"

    if [ -f "$HF_HOME/token" ]; then
        export HF_TOKEN=$(cat "$HF_HOME/token")
    fi

    OUTPUT_BASE="$PROJECT_DIR/results"
fi

# Do NOT set CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

echo "Running limited test (only 2 scales, 20 samples)..."

# Run with minimal configuration
python caa_truthfulqa.py \
    --model_name google/gemma-2-2b \
    --layer 12 \
    --scales 0 1.0 \
    --caa_samples 20 \
    --num_mc_samples 20 \
    --num_gen_samples 0 \
    --judge_model "" \
    --output_dir $OUTPUT_BASE/test_hang

EXIT_CODE=$?
echo ""
echo "Python script exited with code: $EXIT_CODE"
echo "Test complete at $(date)"