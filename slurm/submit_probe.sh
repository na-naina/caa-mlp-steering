#!/bin/bash
# Submit layer probe job
#
# Usage:
#   ./submit_probe.sh google/gemma-3-12b-it
#   ./submit_probe.sh google/gemma-3-12b-it --layers 10 15 20 25 30

set -e

MODEL=${1:?"Usage: ./submit_probe.sh <model_name> [extra args]"}
shift
EXTRA_ARGS="$@"

PROJECT_DIR="/springbrook/share/dcsresearch/u5584851/caa_steering"
SHARE_DIR="/springbrook/share/dcsresearch/u5584851"

mkdir -p "$PROJECT_DIR/logs"

# Clean model name for job name
JOB_NAME="probe_$(echo $MODEL | sed 's|/|_|g' | sed 's|google_||')"

echo "Submitting layer probe: $MODEL"
echo "Job name: $JOB_NAME"

sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=$PROJECT_DIR/logs/${JOB_NAME}_%j.out
#SBATCH --error=$PROJECT_DIR/logs/${JOB_NAME}_%j.err

export HF_HOME="$SHARE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SHARE_DIR/hf_cache/transformers"
export HF_DATASETS_CACHE="$SHARE_DIR/hf_cache/datasets"

echo "=== Layer Probe ==="
echo "Model: $MODEL"
echo "Node: \$(hostname)"
echo "GPU: \$CUDA_VISIBLE_DEVICES"
echo ""

cd $PROJECT_DIR
source venv/bin/activate

python scripts/probe_layers.py --model "$MODEL" $EXTRA_ARGS

echo ""
echo "Done: \$(date)"
SLURM
