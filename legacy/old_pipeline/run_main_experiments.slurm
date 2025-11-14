#!/bin/bash
#SBATCH --job-name=caa_gemma_all
#SBATCH --array=0-35%6  # 36 total experiments, run 6 in parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compute
#SBATCH --output=logs/main_%A_%a.out
#SBATCH --error=logs/main_%A_%a.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

# Activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/$USER/.cache/huggingface  # Use scratch for model cache

# Define main experiments
# Format: MODEL_KEY:LAYER:SCALE_SET:USE_MLP
EXPERIMENTS=(
    # === GEMMA 2 FAMILY ===
    # Gemma2-2B experiments
    "gemma2-2b:8:standard:false"
    "gemma2-2b:12:standard:false"
    "gemma2-2b:16:standard:false"
    "gemma2-2b:12:standard:true"   # With MLP at optimal layer
    "gemma2-2b-it:12:standard:false"
    "gemma2-2b-it:12:standard:true"

    # Gemma2-9B experiments
    "gemma2-9b:18:standard:false"
    "gemma2-9b:21:standard:false"  # Optimal layer
    "gemma2-9b:30:standard:false"
    "gemma2-9b:21:standard:true"   # With MLP at optimal layer
    "gemma2-9b-it:21:standard:false"
    "gemma2-9b-it:21:standard:true"

    # Gemma2-27B experiments (resource intensive)
    "gemma2-27b:16:coarse:false"   # Coarse for large model
    "gemma2-27b:24:coarse:false"   # Optimal layer
    "gemma2-27b:32:coarse:false"
    "gemma2-27b:24:coarse:true"    # With MLP at optimal layer
    "gemma2-27b-it:24:coarse:false"
    "gemma2-27b-it:24:coarse:true"

    # === GEMMA 3 FAMILY ===
    # Gemma3-1B experiments (replacing 2B)
    "gemma3-1b:6:standard:false"
    "gemma3-1b:9:standard:false"   # Optimal layer
    "gemma3-1b:12:standard:false"
    "gemma3-1b:9:standard:true"    # With MLP at optimal layer
    "gemma3-1b-it:9:standard:false"
    "gemma3-1b-it:9:standard:true"

    # Gemma3-9B experiments
    "gemma3-9b:18:standard:false"
    "gemma3-9b:21:standard:false"  # Optimal layer
    "gemma3-9b:30:standard:false"
    "gemma3-9b:21:standard:true"   # With MLP at optimal layer
    "gemma3-9b-it:21:standard:false"
    "gemma3-9b-it:21:standard:true"

    # Gemma3-27B experiments (resource intensive)
    "gemma3-27b:20:coarse:false"
    "gemma3-27b:26:coarse:false"   # Optimal layer
    "gemma3-27b:32:coarse:false"
    "gemma3-27b:26:coarse:true"    # With MLP at optimal layer
    "gemma3-27b-it:26:coarse:false"
    "gemma3-27b-it:26:coarse:true"
)

# Get current experiment configuration
EXP_CONFIG=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
IFS=':' read -r MODEL_KEY LAYER SCALE_SET USE_MLP <<< "$EXP_CONFIG"

# Determine evaluation set based on model size
if [[ "$MODEL_KEY" == *"1b"* ]] || [[ "$MODEL_KEY" == *"2b"* ]]; then
    EVAL_SET="standard"
elif [[ "$MODEL_KEY" == *"9b"* ]]; then
    EVAL_SET="standard"
elif [[ "$MODEL_KEY" == *"27b"* ]]; then
    EVAL_SET="quick"  # Smaller eval set for large models
else
    EVAL_SET="standard"
fi

# Special handling for 27B models
if [[ "$MODEL_KEY" == *"27b"* ]]; then
    echo "Adjusting resources for large model: ${MODEL_KEY}"
    # Request more memory if needed
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
    # Use smaller batch sizes
    BATCH_SIZE_OVERRIDE="2"
else
    BATCH_SIZE_OVERRIDE=""
fi

# Import configurations
python -c "
from model_configs_updated import MODEL_CONFIGS, SCALE_CONFIGS, TRUTHFULQA_CONFIGS
import sys

model = MODEL_CONFIGS.get('${MODEL_KEY}', None)
if model is None:
    print(f'ERROR: Model key ${MODEL_KEY} not found', file=sys.stderr)
    sys.exit(1)

scales = SCALE_CONFIGS['${SCALE_SET}']
eval_cfg = TRUTHFULQA_CONFIGS['${EVAL_SET}']

print(f'MODEL_NAME={model[\"model_name\"]}')
print(f'SCALES={\" \".join(map(str, scales))}')
print(f'NUM_MC={eval_cfg[\"num_mc_samples\"]}')
print(f'NUM_GEN={eval_cfg[\"num_gen_samples\"]}')
print(f'MAX_TOKENS={eval_cfg.get(\"max_new_tokens\", 100)}')
" > temp_config_${SLURM_ARRAY_TASK_ID}.sh

source temp_config_${SLURM_ARRAY_TASK_ID}.sh
rm temp_config_${SLURM_ARRAY_TASK_ID}.sh

# Use Gemma3-27B-IT as judge model (from HuggingFace)
JUDGE_MODEL="google/gemma-3-27b-it"

# For 27B models being evaluated, use a smaller judge or skip
if [[ "$MODEL_KEY" == *"27b"* ]]; then
    JUDGE_MODEL="google/gemma-3-9b-it"  # Use smaller judge for 27B models
fi

# Build command
CMD="python caa_truthfulqa.py \
    --model_name ${MODEL_NAME} \
    --layer ${LAYER} \
    --scales ${SCALES} \
    --caa_samples 100 \
    --num_mc_samples ${NUM_MC} \
    --num_gen_samples ${NUM_GEN} \
    --device cuda \
    --output_dir results/${MODEL_KEY} \
    --judge_model ${JUDGE_MODEL}"

# Add MLP flag if requested
if [ "$USE_MLP" = "true" ]; then
    CMD="${CMD} --use_mlp"
fi

echo "=================================================="
echo "CAA TruthfulQA Main Experiment ${SLURM_ARRAY_TASK_ID}"
echo "=================================================="
echo "Model: ${MODEL_NAME}"
echo "Layer: ${LAYER}"
echo "Scale Set: ${SCALE_SET}"
echo "Eval Set: ${EVAL_SET}"
echo "Use MLP: ${USE_MLP}"
echo "Judge Model: ${JUDGE_MODEL}"
echo "Scales: ${SCALES}"
echo "=================================================="

# Run experiment
$CMD

if [ $? -eq 0 ]; then
    echo "✓ Experiment ${SLURM_ARRAY_TASK_ID} completed successfully"
else
    echo "✗ Experiment ${SLURM_ARRAY_TASK_ID} failed"
    exit 1
fi