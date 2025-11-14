#!/bin/bash
#SBATCH --job-name=setup_cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/setup_cache_%j.out
#SBATCH --error=logs/setup_cache_%j.err

set -euo pipefail

SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
PROJECT_SUBDIR="caa_gemma"
PROJECT_DIR="$SHARED_DIR/$PROJECT_SUBDIR"

mkdir -p "$PROJECT_DIR/cache/huggingface"
mkdir -p "$PROJECT_DIR/cache/transformers"
mkdir -p "$PROJECT_DIR/cache/datasets"
mkdir -p "$PROJECT_DIR/results"

TOKEN_PATH="$HOME/.cache/huggingface/token"
if [ -f "$TOKEN_PATH" ]; then
  cp "$TOKEN_PATH" "$PROJECT_DIR/cache/huggingface/token"
fi

env_setup="$PROJECT_DIR/setup_env.sh"
cat > "$env_setup" <<'ENVEOF'
#!/bin/bash
export SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
export PROJECT_DIR="$SHARED_DIR/caa_gemma"
export HF_HOME="$PROJECT_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/transformers"
export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"
export CAA_OUTPUT_DIR="$PROJECT_DIR/results"
if [ -f "$HF_HOME/token" ]; then
  export HF_TOKEN=$(cat "$HF_HOME/token")
  export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
fi
ENVEOF
chmod +x "$env_setup"

df -h "$SHARED_DIR"

echo "Shared cache ready at $PROJECT_DIR"
