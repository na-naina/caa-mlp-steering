#!/bin/bash
#SBATCH --job-name=caa_array
#SBATCH --partition=gpu
#SBATCH --array=0-6%3  # 7 models, max 3 running at once
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/array_%A_%a.out
#SBATCH --error=slurm_logs/array_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Array of models to test
MODELS=(
    "gemma2-2b"
    "gemma2-2b-it"
    "gemma2-9b"
    "gemma2-9b-it"
    "gemma-2b"
    "gemma-7b"
    "gemma2-27b"
)

# Get model for this array task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Determine resources based on model size
if [[ "$MODEL" == *"27b"* ]]; then
    # Request more resources for 27B model
    echo "Large model detected, may need manual intervention"
    # You might want to submit this separately with more GPUs
    exit 0
elif [[ "$MODEL" == *"9b"* ]] || [[ "$MODEL" == *"7b"* ]]; then
    EVAL_MODE="quick"
    SCALE_GRAN="standard"
else
    EVAL_MODE="standard"
    SCALE_GRAN="standard"
fi

# Create log directory
mkdir -p slurm_logs

# Load modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.2.0
module load cuDNN/8.9.7.29-CUDA-12.2.0

# Setup environment
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets accelerate
    pip install tqdm numpy scipy sentencepiece protobuf huggingface_hub
else
    source venv/bin/activate
fi

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/springbrook/share/dcsresearch/.cache/huggingface

echo "Array Job ${SLURM_ARRAY_JOB_ID} - Task ${SLURM_ARRAY_TASK_ID}"
echo "Running model: ${MODEL}"
echo "Evaluation mode: ${EVAL_MODE}"
echo "Scale granularity: ${SCALE_GRAN}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run the experiment
python batch_runner.py \
    --models ${MODEL} \
    --scale_granularity ${SCALE_GRAN} \
    --eval_mode ${EVAL_MODE} \
    --use_mcp \
    --output_dir results/array_${SLURM_ARRAY_JOB_ID}

echo "Task ${SLURM_ARRAY_TASK_ID} completed at $(date)"
