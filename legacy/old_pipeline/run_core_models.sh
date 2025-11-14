#!/bin/bash

# Core model experiments script
# Tests: Gemma2-2B, Gemma2-9B, Gemma2-27B, Gemma3-1B, Gemma3-9B, Gemma3-27B

echo "================================"
echo "CAA Core Models Experiment Setup"
echo "================================"

# Create necessary directories
mkdir -p logs
mkdir -p results

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Function to run single experiment
run_experiment() {
    local MODEL_KEY=$1
    local MODEL_NAME=$2
    local LAYER=$3
    local USE_MLP=$4
    local SCALE_SET=${5:-"standard"}
    local EVAL_SET=${6:-"standard"}

    echo ""
    echo "Running: $MODEL_KEY (Layer $LAYER, MLP: $USE_MLP)"
    echo "----------------------------------------"

    # Determine judge model based on model size
    if [[ "$MODEL_KEY" == *"27b"* ]]; then
        JUDGE="google/gemma-3-9b-it"  # Use smaller judge for 27B models
        EVAL_SET="quick"  # Smaller evaluation set
    else
        JUDGE="google/gemma-3-27b-it"
    fi

    # Build command
    CMD="python caa_truthfulqa.py \
        --model_name $MODEL_NAME \
        --layer $LAYER \
        --scales 0 0.5 1.0 2.0 5.0 10.0 \
        --caa_samples 100 \
        --num_mc_samples 100 \
        --num_gen_samples 100 \
        --judge_model $JUDGE \
        --output_dir results/$MODEL_KEY"

    if [ "$USE_MLP" = "true" ]; then
        CMD="$CMD --use_mlp"
    fi

    # Run the experiment
    $CMD

    if [ $? -eq 0 ]; then
        echo "✓ $MODEL_KEY completed"
    else
        echo "✗ $MODEL_KEY failed"
    fi
}

# Core experiments list
echo ""
echo "Experiments to run:"
echo "1. Gemma2-2B (base)"
echo "2. Gemma2-9B (base)"
echo "3. Gemma2-27B (base)"
echo "4. Gemma3-1B (replaces 2B)"
echo "5. Gemma3-9B"
echo "6. Gemma3-27B"
echo ""

# === GEMMA 2 FAMILY ===
echo "=== Running Gemma-2 Family ==="

# Gemma2-2B
run_experiment "gemma2-2b" "google/gemma-2-2b" 12 false
run_experiment "gemma2-2b-mlp" "google/gemma-2-2b" 12 true

# Gemma2-9B
run_experiment "gemma2-9b" "google/gemma-2-9b" 21 false
run_experiment "gemma2-9b-mlp" "google/gemma-2-9b" 21 true

# Gemma2-27B (resource intensive - smaller evaluation)
run_experiment "gemma2-27b" "google/gemma-2-27b" 24 false coarse quick
run_experiment "gemma2-27b-mlp" "google/gemma-2-27b" 24 true coarse quick

# === GEMMA 3 FAMILY ===
echo ""
echo "=== Running Gemma-3 Family ==="

# Gemma3-1B (smallest Gemma-3)
run_experiment "gemma3-1b" "google/gemma-3-1b" 9 false
run_experiment "gemma3-1b-mlp" "google/gemma-3-1b" 9 true

# Gemma3-9B
run_experiment "gemma3-9b" "google/gemma-3-9b" 21 false
run_experiment "gemma3-9b-mlp" "google/gemma-3-9b" 21 true

# Gemma3-27B (resource intensive - smaller evaluation)
run_experiment "gemma3-27b" "google/gemma-3-27b" 26 false coarse quick
run_experiment "gemma3-27b-mlp" "google/gemma-3-27b" 26 true coarse quick

echo ""
echo "================================"
echo "All core experiments completed!"
echo "================================"