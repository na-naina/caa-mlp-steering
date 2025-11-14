#!/bin/bash
#SBATCH --job-name=caa_all_models
#SBATCH --array=0-35%4  # Run up to 4 jobs in parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compute
#SBATCH --output=logs/caa_array_%A_%a.out
#SBATCH --error=logs/caa_array_%A_%a.err

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

# Define experiment configurations
# Format: MODEL_KEY:LAYER:SCALE_SET:EVAL_SET:USE_MLP
EXPERIMENTS=(
    # Gemma2 family experiments
    "gemma2-2b:8:standard:standard:false"
    "gemma2-2b:12:standard:standard:false"
    "gemma2-2b:16:standard:standard:false"
    "gemma2-2b:12:standard:standard:true"  # With MLP

    "gemma2-2b-it:8:standard:standard:false"
    "gemma2-2b-it:12:standard:standard:false"
    "gemma2-2b-it:16:standard:standard:false"
    "gemma2-2b-it:12:standard:standard:true"  # With MLP

    "gemma2-9b:18:standard:standard:false"
    "gemma2-9b:24:standard:standard:false"
    "gemma2-9b:30:standard:standard:false"
    "gemma2-9b:21:standard:standard:true"  # With MLP

    "gemma2-9b-it:18:standard:standard:false"
    "gemma2-9b-it:24:standard:standard:false"
    "gemma2-9b-it:21:standard:standard:true"  # With MLP

    # Gemma3 family experiments
    "gemma3-270m:4:standard:standard:false"
    "gemma3-270m:6:standard:standard:false"
    "gemma3-270m:8:standard:standard:false"
    "gemma3-270m:6:standard:standard:true"  # With MLP

    "gemma3-1b:6:standard:standard:false"
    "gemma3-1b:9:standard:standard:false"
    "gemma3-1b:12:standard:standard:false"
    "gemma3-1b:9:standard:standard:true"  # With MLP

    "gemma3-4b:10:standard:standard:false"
    "gemma3-4b:15:standard:standard:false"
    "gemma3-4b:20:standard:standard:false"
    "gemma3-4b:15:standard:standard:true"  # With MLP

    "gemma3-12b:15:standard:thorough:false"
    "gemma3-12b:22:standard:thorough:false"
    "gemma3-12b:30:standard:thorough:false"
    "gemma3-12b:22:standard:thorough:true"  # With MLP

    # Gemma1 for comparison
    "gemma-2b:6:standard:standard:false"
    "gemma-2b:9:standard:standard:false"
    "gemma-2b:9:standard:standard:true"  # With MLP

    "gemma-7b:10:standard:standard:false"
    "gemma-7b:15:standard:standard:false"
    "gemma-7b:15:standard:standard:true"  # With MLP
)

# Get current experiment configuration
EXP_CONFIG=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
IFS=':' read -r MODEL_KEY LAYER SCALE_SET EVAL_SET USE_MLP <<< "$EXP_CONFIG"

# Special handling for large models
if [[ "$MODEL_KEY" == *"27b"* ]]; then
    echo "Adjusting resources for large model: ${MODEL_KEY}"
    export CUDA_VISIBLE_DEVICES=0,1  # Use 2 GPUs for 27B models
    BATCH_SIZE_OVERRIDE="--batch_size 2"
fi

# Import configurations
python -c "
from model_configs_updated import MODEL_CONFIGS, SCALE_CONFIGS, TRUTHFULQA_CONFIGS

model = MODEL_CONFIGS.get('${MODEL_KEY}', None)
if model is None:
    print('ERROR: Model key ${MODEL_KEY} not found')
    exit(1)

scales = SCALE_CONFIGS['${SCALE_SET}']
eval_cfg = TRUTHFULQA_CONFIGS['${EVAL_SET}']

print(f'MODEL_NAME={model[\"model_name\"]}')
print(f'SCALES={\" \".join(map(str, scales))}')
print(f'NUM_MC={eval_cfg[\"num_mc_samples\"]}')
print(f'NUM_GEN={eval_cfg[\"num_gen_samples\"]}')
print(f'MAX_TOKENS={eval_cfg.get(\"max_new_tokens\", 100)}')
" > temp_config.sh

source temp_config.sh
rm temp_config.sh

# Use Gemma3-27B as judge for open-ended evaluation (if available)
JUDGE_MODEL="google/gemma-2-27b"  # Using Gemma2-27B as judge (more likely to be available)

# Build command
CMD="python caa_truthfulqa.py \
    --model_name ${MODEL_NAME} \
    --layer ${LAYER} \
    --scales ${SCALES} \
    --num_mc_samples ${NUM_MC} \
    --num_gen_samples ${NUM_GEN} \
    --device cuda \
    --output_dir results/${MODEL_KEY} \
    ${BATCH_SIZE_OVERRIDE}"

# Add MLP flag if requested
if [ "$USE_MLP" = "true" ]; then
    CMD="${CMD} --use_mlp"
fi

# Add judge model for larger experiments
if [[ "$EVAL_SET" == "thorough" ]] || [[ "$EVAL_SET" == "full" ]]; then
    CMD="${CMD} --judge_model ${JUDGE_MODEL}"
fi

echo "=================================================="
echo "CAA TruthfulQA Array Job ${SLURM_ARRAY_TASK_ID}"
echo "=================================================="
echo "Model: ${MODEL_NAME}"
echo "Layer: ${LAYER}"
echo "Scale Set: ${SCALE_SET}"
echo "Eval Set: ${EVAL_SET}"
echo "Use MLP: ${USE_MLP}"
echo "Judge Model: ${JUDGE_MODEL}"
echo "=================================================="

# Run experiment
$CMD

echo "Experiment ${SLURM_ARRAY_TASK_ID} completed for ${MODEL_KEY} at layer ${LAYER}"