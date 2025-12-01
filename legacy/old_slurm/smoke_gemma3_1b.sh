#!/bin/bash
#SBATCH --job-name=smoke_gemma3_1b
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:lovelace_l40:2
#SBATCH --output=logs/smoke_gemma3_1b_%j.out
#SBATCH --error=logs/smoke_gemma3_1b_%j.err

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

python -m src.jobs.run_experiment --base-config configs/base.yaml --config configs/smoke_gemma3_1b.yaml --run-id smoke_gemma3_1b_$(date +%Y%m%d_%H%M%S)
