#!/bin/bash
# CAA Steering Job Submission
# Usage: ./submit.sh <model_name> [stage]

set -e

MODEL=${1:?"Usage: ./submit.sh <model_name> [stage]"}
STAGE=${2:-all}

PROJECT_DIR="/springbrook/share/dcsresearch/u5584851/caa_steering"
SHARE_DIR="/springbrook/share/dcsresearch/u5584851"
CONFIG_FILE="$PROJECT_DIR/configs/models/$MODEL.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config not found: $CONFIG_FILE"
    exit 1
fi

# Extract SLURM settings from config
GPUS=$(grep -A5 '^slurm:' "$CONFIG_FILE" | grep 'gpus:' | awk '{print $2}' || echo 1)
MEM=$(grep -A5 '^slurm:' "$CONFIG_FILE" | grep 'mem_gb:' | awk '{print $2}' || echo 80)
TIME=$(grep -A5 '^slurm:' "$CONFIG_FILE" | grep 'time:' | awk '{print $2}' | tr -d '"' || echo "08:00:00")

GPUS=${GPUS:-1}
MEM=${MEM:-80}
TIME=${TIME:-08:00:00}
CPUS=$((GPUS * 10))
JOB_NAME="caa_${MODEL}"

echo "Submitting: $JOB_NAME (stage=$STAGE)"
echo "  GPUs: $GPUS, Mem: ${MEM}G, Time: $TIME"

sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=${MEM}G
#SBATCH --time=$TIME
#SBATCH --output=$PROJECT_DIR/logs/${JOB_NAME}_%j.out
#SBATCH --error=$PROJECT_DIR/logs/${JOB_NAME}_%j.err

# Use shared storage for caches (avoid home quota)
export HF_HOME="$SHARE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SHARE_DIR/hf_cache/transformers"
export HF_DATASETS_CACHE="$SHARE_DIR/hf_cache/datasets"

echo "Job started: \$(date)"
echo "Node: \$(hostname)"
echo "GPUs: \$CUDA_VISIBLE_DEVICES"
echo "HF_HOME: \$HF_HOME"

cd $PROJECT_DIR
source venv/bin/activate

python run.py --model $MODEL --stage $STAGE --verbose

echo "Job finished: \$(date)"
SLURM
