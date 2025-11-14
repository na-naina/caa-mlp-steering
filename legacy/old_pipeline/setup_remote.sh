#!/bin/bash

# Initial setup script for remote SLURM server
# Usage: ./setup_remote.sh username@server /path/to/project

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ $# -ne 2 ]; then
    echo "Usage: $0 username@server /path/to/remote/project"
    exit 1
fi

SERVER=$1
REMOTE_PATH=$2

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Remote SLURM Environment Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 1: Create remote directory
echo -e "${YELLOW}Creating remote directory...${NC}"
ssh ${SERVER} "mkdir -p ${REMOTE_PATH}"

# Step 2: Transfer all files
echo -e "${YELLOW}Transferring project files...${NC}"
rsync -avz --progress \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    ./ ${SERVER}:${REMOTE_PATH}/

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to transfer files${NC}"
    exit 1
fi

# Step 3: Setup Python environment
echo -e "${YELLOW}Setting up Python environment...${NC}"
ssh ${SERVER} << 'ENDSSH'
cd ${REMOTE_PATH}

# Load modules
echo "Loading modules..."
module load Python/3.10.4 2>/dev/null || module load python/3.10 2>/dev/null || {
    echo "Warning: Could not load Python module. Available modules:"
    module avail 2>&1 | grep -i python
}

module load CUDA/12.1.1 2>/dev/null || module load cuda/12.1 2>/dev/null || {
    echo "Warning: Could not load CUDA module. Available modules:"
    module avail 2>&1 | grep -i cuda
}

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p results

# Set up HuggingFace cache in scratch
if [ -d "/scratch/$USER" ]; then
    mkdir -p /scratch/$USER/.cache/huggingface
    ln -sf /scratch/$USER/.cache/huggingface ~/.cache/huggingface
    echo "HuggingFace cache linked to scratch directory"
fi

echo ""
echo "Environment setup complete!"
echo ""
echo "Python version:"
python --version
echo ""
echo "Installed packages:"
pip list | grep -E "torch|transformers|datasets"
ENDSSH

# Step 4: Test the setup
echo ""
echo -e "${YELLOW}Testing setup...${NC}"
ssh ${SERVER} << ENDTEST
cd ${REMOTE_PATH}
source venv/bin/activate

# Quick import test
python -c "
import torch
import transformers
from datasets import load_dataset
print('✓ PyTorch version:', torch.__version__)
print('✓ Transformers version:', transformers.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU:', torch.cuda.get_device_name(0))
"
ENDTEST

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Submit jobs:"
echo "   ./remote_submit.sh ${SERVER} ${REMOTE_PATH}"
echo ""
echo "2. Or SSH directly:"
echo "   ssh ${SERVER}"
echo "   cd ${REMOTE_PATH}"
echo "   source venv/bin/activate"
echo "   python submit_core_experiments.py"