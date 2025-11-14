#!/bin/bash
#SBATCH --job-name=caa_gemma_large
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/caa_large_%j.out
#SBATCH --error=slurm_logs/caa_large_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Load necessary modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.2.0
module load cuDNN/8.9.7.29-CUDA-12.2.0

# Activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets accelerate
    pip install tqdm numpy scipy
    pip install sentencepiece protobuf
    pip install huggingface_hub bitsandbytes
else
    source venv/bin/activate
fi

# Set environment variables for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false

# For larger models, we may need to use model parallelism
export ACCELERATE_USE_FSDP=1
export FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP

# Model to run
MODEL=${1:-"gemma2-9b"}
SCALE_GRAN=${2:-"standard"}
EVAL_MODE=${3:-"quick"}  # Use quick for large models initially

echo "Starting CAA experiment for large model: ${MODEL}"
echo "Scale granularity: ${SCALE_GRAN}"
echo "Evaluation mode: ${EVAL_MODE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Run experiment with special handling for large models
python batch_runner.py \
    --models ${MODEL} \
    --scale_granularity ${SCALE_GRAN} \
    --eval_mode ${EVAL_MODE} \
    --use_mcp \
    --output_dir results/slurm_${SLURM_JOB_ID}

echo "Experiment completed at $(date)"
