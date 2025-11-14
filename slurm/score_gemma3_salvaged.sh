#!/bin/bash
#SBATCH --job-name=score_gemma3_salvaged
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:lovelace_l40:2
#SBATCH --output=logs/score_gemma3_salvaged_%j.out
#SBATCH --error=logs/score_gemma3_salvaged_%j.err

set -euo pipefail

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

# Score all salvageable experiments
for dir in outputs/gemma3_270m_it_20251114_125537 outputs/gemma3_1b_it_20251114_125537 outputs/gemma3_4b_it_20251114_130117 outputs/gemma3_12b_2gpu_20251114_124809; do
    echo "===== Scoring $dir ====="
    python scripts/score_generated_responses.py "$dir" \
        --judge-model google/gemma-3-12b-it \
        --judge-device cuda:1 \
        --no-bleurt
done

echo "All salvaged experiments scored!"
