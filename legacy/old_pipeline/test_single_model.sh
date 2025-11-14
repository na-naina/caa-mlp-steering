#!/bin/bash
#SBATCH --job-name=test_gemma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --partition=gpu
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

echo "=================================================="
echo "Test Single Model Experiment"
echo "=================================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo ""

# Load modules
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

# Activate virtual environment
source venv/bin/activate

# Setup environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
export TOKENIZERS_PARALLELISM=false

# Setup shared storage
SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
PROJECT_DIR="$SHARED_DIR/caa_experiments"

if [ -d "$SHARED_DIR" ]; then
    echo "Using shared storage at $SHARED_DIR"
    mkdir -p "$PROJECT_DIR/cache/huggingface"
    mkdir -p "$PROJECT_DIR/cache/transformers"
    mkdir -p "$PROJECT_DIR/results"

    export HF_HOME="$PROJECT_DIR/cache/huggingface"
    export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/transformers"
    export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"
    export OUTPUT_DIR="$PROJECT_DIR/results"

    if [ -f ~/.cache/huggingface/token ] && [ ! -f "$HF_HOME/token" ]; then
        cp ~/.cache/huggingface/token "$HF_HOME/token"
    fi
    if [ -f "$HF_HOME/token" ]; then
        export HF_TOKEN=$(cat "$HF_HOME/token")
    fi
else
    echo "Using local cache (shared dir not accessible)"
    export HF_HOME=$PWD/.cache/huggingface
    export TRANSFORMERS_CACHE=$PWD/.cache/transformers
    export OUTPUT_DIR=$PWD/results
    mkdir -p $HF_HOME
    mkdir -p $TRANSFORMERS_CACHE

    if [ -f ~/.cache/huggingface/token ]; then
        cp ~/.cache/huggingface/token $HF_HOME/token
        export HF_TOKEN=$(cat $HF_HOME/token)
    fi
fi

echo "Environment setup complete"
echo ""

# Run a minimal test - gemma-2b with minimal settings
echo "Running minimal CAA experiment..."
python -c "
import torch
import os
import sys

# Set cache
os.environ['HF_HOME'] = '$HF_HOME'

print('1. Testing imports...')
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print('2. Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
print('   ✓ Tokenizer loaded')

print('3. Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-2b',
    torch_dtype=torch.float16,
    device_map='auto'
)
print('   ✓ Model loaded')

print('4. Loading dataset...')
dataset = load_dataset('truthful_qa', 'generation', split='validation')
print(f'   ✓ Dataset loaded: {len(dataset)} samples')

print('5. Testing inference...')
inputs = tokenizer('Hello world', return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'   ✓ Inference works: {result[:50]}...')

print('')
print('✓ All tests passed! Ready for full experiments.')
"

# If test passes, run minimal CAA experiment
if [ $? -eq 0 ]; then
    echo ""
    echo "Running minimal CAA experiment..."
    python caa_truthfulqa.py \
        --model_name google/gemma-2b \
        --layer 6 \
        --scales 0 1.0 \
        --caa_samples 20 \
        --num_mc_samples 10 \
        --num_gen_samples 10 \
        --judge_model "" \
        --output_dir ${OUTPUT_DIR:-test_results}
fi

echo ""
echo "Test complete!"