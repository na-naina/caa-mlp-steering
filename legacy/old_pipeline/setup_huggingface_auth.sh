#!/bin/bash

echo "========================================"
echo "HuggingFace Authentication Setup"
echo "========================================"
echo ""

echo "This script will help you authenticate with HuggingFace to access Gemma models."
echo ""
echo "Prerequisites:"
echo "1. Create a HuggingFace account at https://huggingface.co/join"
echo "2. Accept the Gemma model licenses:"
echo "   - https://huggingface.co/google/gemma-2-2b"
echo "   - https://huggingface.co/google/gemma-2-9b"
echo "   - https://huggingface.co/google/gemma-2-27b"
echo "   - https://huggingface.co/google/gemma-3-1b"
echo "   - https://huggingface.co/google/gemma-3-9b"
echo "   - https://huggingface.co/google/gemma-3-27b"
echo "3. Create an access token at https://huggingface.co/settings/tokens"
echo ""
echo "Press Enter when you have completed these steps..."
read

echo "Please enter your HuggingFace access token:"
read -s HF_TOKEN

echo ""
echo "Setting up authentication..."

# Method 1: Using huggingface-cli
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli login..."
    huggingface-cli login --token $HF_TOKEN
else
    echo "huggingface-cli not found, using alternative method..."
fi

# Method 2: Set environment variable
echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
echo "export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" >> ~/.bashrc

# Method 3: Create token file
mkdir -p ~/.cache/huggingface
echo -n $HF_TOKEN > ~/.cache/huggingface/token

# Method 4: Git credential store for HF
git config --global credential.helper store

echo ""
echo "Authentication setup complete!"
echo ""
echo "The token has been saved in multiple locations:"
echo "1. ~/.cache/huggingface/token"
echo "2. Environment variables in ~/.bashrc"
echo ""
echo "Now you can run the experiments with authentication."
echo ""
echo "To test authentication, run:"
echo "  source ~/.bashrc"
echo "  python -c \"from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('google/gemma-2-2b'); print('Success!')\""