#!/bin/bash
#SBATCH --job-name=setup_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --partition=gpu
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

echo "=================================================="
echo "Setting up Shared Storage for Experiments"
echo "=================================================="
echo ""

# Define shared folder path
SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
PROJECT_NAME="caa_experiments"
SHARED_PROJECT_DIR="$SHARED_DIR/$PROJECT_NAME"

# Check if shared directory is accessible
echo "Checking shared directory access..."
if [ ! -d "$SHARED_DIR" ]; then
    echo "ERROR: Shared directory $SHARED_DIR not accessible!"
    exit 1
fi

echo "✓ Shared directory accessible"
echo ""

# Create project directory in shared space
echo "Creating project directory in shared space..."
mkdir -p "$SHARED_PROJECT_DIR"
mkdir -p "$SHARED_PROJECT_DIR/cache/huggingface"
mkdir -p "$SHARED_PROJECT_DIR/cache/transformers"
mkdir -p "$SHARED_PROJECT_DIR/results"
mkdir -p "$SHARED_PROJECT_DIR/models"

echo "✓ Created directories:"
echo "  - $SHARED_PROJECT_DIR/cache/"
echo "  - $SHARED_PROJECT_DIR/results/"
echo "  - $SHARED_PROJECT_DIR/models/"
echo ""

# Copy HF token if exists
if [ -f ~/.cache/huggingface/token ]; then
    cp ~/.cache/huggingface/token "$SHARED_PROJECT_DIR/cache/huggingface/token"
    echo "✓ Copied HuggingFace token"
fi

# Check disk space
echo ""
echo "Disk space in shared directory:"
df -h "$SHARED_DIR"
echo ""

# Create environment setup script
cat > "$SHARED_PROJECT_DIR/setup_env.sh" << 'EOF'
#!/bin/bash
# Source this file to set up environment variables

export SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
export PROJECT_DIR="$SHARED_DIR/caa_experiments"

# Set cache directories
export HF_HOME="$PROJECT_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/transformers"
export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"

# Set output directory
export CAA_OUTPUT_DIR="$PROJECT_DIR/results"

# Load token
if [ -f "$HF_HOME/token" ]; then
    export HF_TOKEN=$(cat "$HF_HOME/token")
fi

echo "Environment configured:"
echo "  HF_HOME=$HF_HOME"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  CAA_OUTPUT_DIR=$CAA_OUTPUT_DIR"
EOF

chmod +x "$SHARED_PROJECT_DIR/setup_env.sh"

echo "✓ Created environment setup script at:"
echo "  $SHARED_PROJECT_DIR/setup_env.sh"
echo ""

# Test write permissions
echo "Testing write permissions..."
echo "Test file created at $(date)" > "$SHARED_PROJECT_DIR/test_write.txt"
if [ -f "$SHARED_PROJECT_DIR/test_write.txt" ]; then
    echo "✓ Write permissions confirmed"
    rm "$SHARED_PROJECT_DIR/test_write.txt"
else
    echo "✗ Cannot write to shared directory!"
    exit 1
fi

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Shared project directory: $SHARED_PROJECT_DIR"
echo ""
echo "To use in SLURM scripts, add:"
echo "  source $SHARED_PROJECT_DIR/setup_env.sh"
echo ""
echo "Python scripts will use:"
echo "  - Cache: $SHARED_PROJECT_DIR/cache/"
echo "  - Results: $SHARED_PROJECT_DIR/results/"