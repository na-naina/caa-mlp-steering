#!/bin/bash
#SBATCH --job-name=predownload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/predownload_%j.out
#SBATCH --error=logs/predownload_%j.err

echo "=================================================="
echo "Pre-downloading Models to Shared Cache"
echo "=================================================="
echo ""

module purge
module load Python/3.10.4-GCCcore-11.3.0

source venv/bin/activate

# Setup shared storage
SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
PROJECT_DIR="$SHARED_DIR/caa_experiments"

mkdir -p "$PROJECT_DIR/cache/huggingface"
export HF_HOME="$PROJECT_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/transformers"

if [ -f ~/.cache/huggingface/token ]; then
    cp ~/.cache/huggingface/token "$HF_HOME/token"
    export HF_TOKEN=$(cat "$HF_HOME/token")
fi

echo "Cache directory: $HF_HOME"
echo ""

# Python script to download models
python << 'EOF'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

models_to_download = [
    # Small models first
    ("google/gemma-2b", "Gemma 2B (smallest)"),
    ("google/gemma-2-2b", "Gemma2 2B"),

    # Medium models
    ("google/gemma-2-9b", "Gemma2 9B"),

    # Large models (optional - comment out if not needed)
    # ("google/gemma-2-27b", "Gemma2 27B"),
    # ("google/gemma-3-27b-it", "Gemma3 27B-IT (Judge)"),
]

print("Starting model downloads...\n")

for model_name, description in models_to_download:
    print(f"Downloading {description}: {model_name}")
    print("-" * 50)

    try:
        # Download config first
        print("  1. Downloading config...")
        config = AutoConfig.from_pretrained(model_name)

        # Download tokenizer
        print("  2. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Download model weights (but don't load into memory)
        print("  3. Downloading model weights...")
        # This downloads but doesn't load the full model into RAM
        model_path = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # Only download, don't load
            local_files_only=False,
            cache_dir=os.environ.get('HF_HOME'),
        )

        # Immediately delete from memory
        del model_path

        print(f"  ✓ Successfully downloaded {description}\n")

    except Exception as e:
        print(f"  ✗ Error downloading {description}: {e}\n")
        continue

print("\n" + "="*50)
print("Download Summary:")
print("="*50)

# Check cache size
import subprocess
result = subprocess.run(['du', '-sh', os.environ.get('HF_HOME', '.')],
                       capture_output=True, text=True)
print(f"Cache size: {result.stdout.strip()}")

print("\nModels are now cached and ready for experiments!")
EOF

echo ""
echo "=================================================="
echo "Pre-download Complete"
echo "=================================================="