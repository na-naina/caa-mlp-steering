#!/bin/bash

echo "=================================================="
echo "Fixing HuggingFace Cache Setup"
echo "=================================================="
echo ""

# Check if scratch directory is accessible
echo "1. Checking scratch directory access:"
if [ -d "/scratch/$USER" ]; then
    echo "   ✓ /scratch/$USER exists"
    if [ -w "/scratch/$USER" ]; then
        echo "   ✓ /scratch/$USER is writable"
    else
        echo "   ✗ /scratch/$USER is not writable"
    fi
else
    echo "   ✗ /scratch/$USER does not exist"
    echo "   Trying to create it..."
    mkdir -p "/scratch/$USER" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   ✓ Created /scratch/$USER"
    else
        echo "   ✗ Cannot create /scratch/$USER (permission denied)"
        echo "   Will use local cache instead"
    fi
fi

echo ""
echo "2. Setting up local cache directory:"
LOCAL_CACHE="$PWD/.cache"
mkdir -p "$LOCAL_CACHE/huggingface"
mkdir -p "$LOCAL_CACHE/transformers"
echo "   ✓ Created $LOCAL_CACHE"

echo ""
echo "3. Copying HuggingFace token:"
if [ -f ~/.cache/huggingface/token ]; then
    cp ~/.cache/huggingface/token "$LOCAL_CACHE/huggingface/token"
    echo "   ✓ Token copied to local cache"
else
    echo "   ✗ No token found at ~/.cache/huggingface/token"
fi

echo ""
echo "4. Creating .env file for environment variables:"
cat > .env << EOF
# HuggingFace cache configuration
export HF_HOME=$LOCAL_CACHE/huggingface
export TRANSFORMERS_CACHE=$LOCAL_CACHE/transformers
export HUGGINGFACE_HUB_CACHE=$LOCAL_CACHE/huggingface/hub

# Load token if available
if [ -f ~/.cache/huggingface/token ]; then
    export HF_TOKEN=\$(cat ~/.cache/huggingface/token)
    export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN
fi
EOF

echo "   ✓ Created .env file"

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To use the local cache, run:"
echo "  source .env"
echo ""
echo "Models will be cached in:"
echo "  $LOCAL_CACHE"
echo ""
echo "Note: Models can be large (2-50GB each), ensure you have enough space:"
df -h "$PWD" | tail -1