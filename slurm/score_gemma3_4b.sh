#!/bin/bash
#SBATCH --job-name=score_gemma3_4b
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --output=logs/score_gemma3_4b_%j.out
#SBATCH --error=logs/score_gemma3_4b_%j.err

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

# Score pre-generated responses with judge model
python scripts/score_generated_responses.py \
  outputs/run_20251107_172642 \
  --judge-model google/gemma-3-12b-it \
  --judge-device auto
