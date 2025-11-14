#!/bin/bash
#SBATCH --job-name=test_minimal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --partition=gpu
#SBATCH --output=logs/test_minimal_%j.out
#SBATCH --error=logs/test_minimal_%j.err

echo "=================================================="
echo "Minimal Test - Gemma 2B"
echo "=================================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo ""

# Load modules
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

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

    if [ -f ~/.cache/huggingface/token ]; then
        cp ~/.cache/huggingface/token "$HF_HOME/token" 2>/dev/null
    fi
    if [ -f "$HF_HOME/token" ]; then
        export HF_TOKEN=$(cat "$HF_HOME/token")
    fi
fi

# Activate venv
source venv/bin/activate

# Set GPU environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Check GPU
echo "GPU Check:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Create a minimal test script
cat > test_minimal.py << 'EOF'
import torch
import os
import gc

print("Python test starting...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test basic imports
print("\n1. Testing imports...")
from transformers import AutoTokenizer, AutoModelForCausalLM

print("2. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
print("   ✓ Tokenizer loaded")

print("3. Loading model with CPU first...")
try:
    # Load on CPU first to avoid device_map issues
    model = AutoModelForCausalLM.from_pretrained(
        'google/gemma-2b',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None  # Load on CPU
    )
    print("   ✓ Model loaded on CPU")

    print("4. Moving to GPU...")
    model = model.cuda()
    print("   ✓ Model on GPU")

    print("5. Testing generation...")
    inputs = tokenizer("Hello", return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   ✓ Generation works: '{result}'")

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n✓ All tests passed!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

# Run the test
python test_minimal.py

# Clean up
rm -f test_minimal.py

echo ""
echo "=================================================="
echo "Test Complete"
echo "===================================================="