#!/bin/bash
#SBATCH --job-name=gemma3_12b_genonly
#SBATCH --partition=gpu
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --output=logs/gemma3_12b_genonly_%j.out
#SBATCH --error=logs/gemma3_12b_genonly_%j.err

set -euo pipefail

# Module loading and environment setup
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

source /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma
export HF_HOME=/springbrook/share/dcsresearch/u5584851/experiments/caa_gemma/cache/transformers

cd /springbrook/share/dcsresearch/u5584851/experiments/caa_gemma

# Generation only - no judge evaluation
python -m src.jobs.run_experiment --base-config configs/base.yaml --config configs/gemma3_12b_genonly.yaml --run-id gemma3_12b_genonly_$(date +%Y%m%d_%H%M%S)
