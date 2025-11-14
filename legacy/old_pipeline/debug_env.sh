#!/bin/bash
#SBATCH --job-name=debug_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --partition=gpu
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

echo "=================================================="
echo "Debug Environment"
echo "=================================================="
echo ""

# Basic info
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "User: $USER"
echo "PWD: $(pwd)"
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi
echo ""

# Load modules
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

# Activate venv
source venv/bin/activate

# Check Python environment
echo "Python Environment:"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""

# Check PyTorch and CUDA
python -c "
import sys
print('Python:', sys.version)
print()

try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        print('GPU count:', torch.cuda.device_count())
        print('GPU name:', torch.cuda.get_device_name(0))

        # Test basic CUDA operations
        print()
        print('Testing basic CUDA ops...')
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = x @ y
        print('  Matrix multiply: OK')

        # Test memory
        print()
        print('GPU Memory:')
        print(f'  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
        print(f'  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB')
        print(f'  Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

except Exception as e:
    print(f'Error: {e}')

print()
try:
    import transformers
    print('Transformers version:', transformers.__version__)
except:
    print('Transformers not found')
"

echo ""
echo "Testing model load without device_map..."
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print('Loading model on CPU first...')
model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-2b',
    torch_dtype=torch.float16,
    device_map=None  # Load on CPU first
)
print('Model loaded on CPU')

print('Moving to CUDA...')
model = model.cuda()
print('Model on CUDA')

print('Testing generation...')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
inputs = tokenizer('Hello', return_tensors='pt').to('cuda')

with torch.no_grad():
    # Use simpler generation
    output = model(**inputs)
    print('Forward pass OK')

    # Try generation with minimal settings
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Generation OK: {result}')

print('✓ All tests passed!')
"

echo ""
echo "=================================================="
echo "Debug complete"
echo "=================================================="