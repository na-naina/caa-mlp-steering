#!/bin/bash
#SBATCH --job-name=caa_gemma_small
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/caa_%j.out
#SBATCH --error=slurm_logs/caa_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Load necessary modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.2.0
module load cuDNN/8.9.7.29-CUDA-12.2.0

# Activate virtual environment (create if doesn't exist)
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets accelerate
    pip install tqdm numpy scipy
    pip install sentencepiece protobuf
    pip install huggingface_hub
else
    source venv/bin/activate
fi

# Set environment variables for better GPU performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Model to run (can be passed as argument)
MODEL=${1:-"gemma2-2b"}
SCALE_GRAN=${2:-"standard"}
EVAL_MODE=${3:-"standard"}

echo "Starting CAA experiment for ${MODEL}"
echo "Scale granularity: ${SCALE_GRAN}"
echo "Evaluation mode: ${EVAL_MODE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run experiment
python batch_runner.py \
    --models ${MODEL} \
    --scale_granularity ${SCALE_GRAN} \
    --eval_mode ${EVAL_MODE} \
    --use_mcp \
    --output_dir results/slurm_${SLURM_JOB_ID}

echo "Experiment completed at $(date)"
