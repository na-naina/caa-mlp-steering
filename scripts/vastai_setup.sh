#!/bin/bash
# Vast.ai setup script for Llama2-7B-chat experiments
# Run this after SSHing into your instance

set -e

echo "=== Setting up environment ==="

# 1. Clone the repo (or upload your local copy)
# Option A: Clone from git
# git clone https://github.com/YOUR_USERNAME/caa-mlp-steering.git
# cd caa-mlp-steering

# Option B: Already uploaded via scp (recommended)
cd /workspace/caa-mlp-steering

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch (adjust CUDA version if needed)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install requirements
pip install -r requirements.txt

# 5. Authenticate with HuggingFace (required for Llama2)
echo ""
echo "=== HuggingFace Authentication ==="
echo "Llama2 is a gated model. You need to:"
echo "1. Go to https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
echo "2. Accept the license agreement"
echo "3. Create an access token at https://huggingface.co/settings/tokens"
echo ""
read -p "Enter your HuggingFace token: " HF_TOKEN
huggingface-cli login --token "$HF_TOKEN"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the full pipeline:"
echo "  python run.py --model llama2_7b_chat"
echo ""
echo "Or run stages separately:"
echo "  python run.py --model llama2_7b_chat --stage train"
echo "  python run.py --model llama2_7b_chat --stage eval --run-dir outputs/llama2_7b_chat_XXXXXXXX"
