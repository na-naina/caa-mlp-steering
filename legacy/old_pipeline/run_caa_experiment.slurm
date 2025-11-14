#!/bin/bash
#SBATCH --job-name=caa_truthfulqa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compute
#SBATCH --output=logs/caa_%A_%a.out
#SBATCH --error=logs/caa_%A_%a.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

# Setup shared storage environment
SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
PROJECT_DIR="$SHARED_DIR/caa_experiments"

# Check if shared directory is accessible
if [ -d "$SHARED_DIR" ]; then
    echo "Using shared directory: $SHARED_DIR"

    # Create directories if they don't exist
    mkdir -p "$PROJECT_DIR/cache/huggingface"
    mkdir -p "$PROJECT_DIR/cache/transformers"
    mkdir -p "$PROJECT_DIR/results"

    # Set cache directories
    export HF_HOME="$PROJECT_DIR/cache/huggingface"
    export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/transformers"
    export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"

    # Copy token if exists and not already in shared cache
    if [ -f ~/.cache/huggingface/token ] && [ ! -f "$HF_HOME/token" ]; then
        cp ~/.cache/huggingface/token "$HF_HOME/token"
    fi

    # Load token
    if [ -f "$HF_HOME/token" ]; then
        export HF_TOKEN=$(cat "$HF_HOME/token")
    fi

    echo "Cache directories set to shared storage"

    # Set output directory to shared storage
    export OUTPUT_DIR="$PROJECT_DIR/results"
else
    echo "WARNING: Shared directory not accessible, using local cache"
    export HF_HOME="$PWD/.cache/huggingface"
    export TRANSFORMERS_CACHE="$PWD/.cache/transformers"
    export OUTPUT_DIR="$PWD/results"
    mkdir -p "$HF_HOME"
    mkdir -p "$TRANSFORMERS_CACHE"
fi

# Activate virtual environment (create if doesn't exist)
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Parse arguments
MODEL_KEY=${1:-"gemma2-2b"}
LAYER=${2:-12}
SCALE_SET=${3:-"standard"}
EVAL_SET=${4:-"standard"}
USE_MLP=${5:-"true"}
JUDGE_MODEL=${6:-"google/gemma-3-27b-it"}  # Default to HuggingFace Gemma3-27b-it

# Import configurations
python -c "
from model_configs_updated import MODEL_CONFIGS, SCALE_CONFIGS, TRUTHFULQA_CONFIGS

model = MODEL_CONFIGS['${MODEL_KEY}']
scales = SCALE_CONFIGS['${SCALE_SET}']
eval_cfg = TRUTHFULQA_CONFIGS['${EVAL_SET}']

print(f'MODEL_NAME={model[\"model_name\"]}')
print(f'SCALES={\" \".join(map(str, scales))}')
print(f'NUM_MC={eval_cfg[\"num_mc_samples\"]}')
print(f'NUM_GEN={eval_cfg[\"num_gen_samples\"]}')
" > temp_config.sh

source temp_config.sh
rm temp_config.sh

# Build command
CMD="python caa_truthfulqa.py \
    --model_name ${MODEL_NAME} \
    --layer ${LAYER} \
    --scales ${SCALES} \
    --num_mc_samples ${NUM_MC} \
    --num_gen_samples ${NUM_GEN} \
    --device cuda \
    --output_dir ${OUTPUT_DIR:-results}/${MODEL_KEY}"

# Add MLP flag if requested
if [ "$USE_MLP" = "true" ]; then
    CMD="${CMD} --use_mlp"
fi

# Add judge model if provided
if [ -n "$JUDGE_MODEL" ]; then
    CMD="${CMD} --judge_model ${JUDGE_MODEL}"
fi

echo "=================================================="
echo "CAA TruthfulQA Experiment"
echo "=================================================="
echo "Model: ${MODEL_NAME}"
echo "Layer: ${LAYER}"
echo "Scales: ${SCALES}"
echo "MC Samples: ${NUM_MC}"
echo "Gen Samples: ${NUM_GEN}"
echo "Use MLP: ${USE_MLP}"
echo "Judge Model: ${JUDGE_MODEL}"
echo "=================================================="

# Run experiment
$CMD

echo "Experiment completed for ${MODEL_KEY} at layer ${LAYER}"