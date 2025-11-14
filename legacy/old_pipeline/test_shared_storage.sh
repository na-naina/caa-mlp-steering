#!/bin/bash
#SBATCH --job-name=test_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --partition=gpu
#SBATCH --output=logs/test_shared_%j.out
#SBATCH --error=logs/test_shared_%j.err

echo "=================================================="
echo "Testing Shared Storage with Minimal Model"
echo "=================================================="
echo ""

# Setup environment
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

# Setup shared storage
SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
PROJECT_DIR="$SHARED_DIR/caa_experiments"

echo "1. Checking shared directory access..."
if [ ! -d "$SHARED_DIR" ]; then
    echo "ERROR: Shared directory not accessible!"
    exit 1
fi

echo "✓ Shared directory accessible: $SHARED_DIR"
echo ""

# Check disk space
echo "2. Disk space in shared directory:"
df -h "$SHARED_DIR"
echo ""

# Create project directories
echo "3. Setting up project directories..."
mkdir -p "$PROJECT_DIR/cache/huggingface"
mkdir -p "$PROJECT_DIR/cache/transformers"
mkdir -p "$PROJECT_DIR/results"

# Set environment
export HF_HOME="$PROJECT_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/transformers"
export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"

# Copy token
if [ -f ~/.cache/huggingface/token ]; then
    cp ~/.cache/huggingface/token "$HF_HOME/token"
    export HF_TOKEN=$(cat "$HF_HOME/token")
    echo "✓ HuggingFace token configured"
fi

# Activate venv
source venv/bin/activate

# Test Python environment
echo ""
echo "4. Python environment:"
which python
python --version
echo ""

# Test minimal download
echo "5. Testing model download to shared cache..."
python -c "
import os
import sys

# Verify environment
print('Environment variables:')
print(f'  HF_HOME: {os.environ.get(\"HF_HOME\")}')
print(f'  TRANSFORMERS_CACHE: {os.environ.get(\"TRANSFORMERS_CACHE\")}')
print()

try:
    print('Testing tokenizer download...')
    from transformers import AutoTokenizer

    # Try smallest model first
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
    print('✓ Successfully downloaded gemma-2b tokenizer')

    # Check cache location
    import pathlib
    cache_dir = pathlib.Path(os.environ.get('HF_HOME', ''))
    if cache_dir.exists():
        files = list(cache_dir.rglob('*'))
        print(f'✓ Cache created at: {cache_dir}')
        print(f'  Files in cache: {len(files)}')

    print()
    print('✓ Test successful! Ready to run experiments with shared storage.')

except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "=================================================="
echo "Test Complete"
echo "=================================================="