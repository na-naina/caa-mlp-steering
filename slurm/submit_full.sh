#!/bin/bash
# Submit full pipeline with automatic run_id propagation
#
# This script submits all stages and handles run_id passing between them.
# The extract stage writes the run_id to a file, subsequent stages read it.
#
# Usage:
#   ./submit_full.sh <model_name>

set -e

MODEL=${1:?"Usage: ./submit_full.sh <model_name>"}

PROJECT_DIR="/springbrook/share/dcsresearch/u5584851/caa_steering"
SHARE_DIR="/springbrook/share/dcsresearch/u5584851"
CONFIG_FILE="$PROJECT_DIR/configs/models/$MODEL.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config not found: $CONFIG_FILE"
    exit 1
fi

mkdir -p "$PROJECT_DIR/logs"

# Extract config values
get_config_value() {
    grep -A10 "^slurm:" "$CONFIG_FILE" | grep "$1:" | awk '{print $2}' | tr -d '"' || echo ""
}

TRAIN_GPUS=$(get_config_value "gpus")
TRAIN_MEM=$(get_config_value "mem_gb")

TRAIN_GPUS=${TRAIN_GPUS:-2}
TRAIN_MEM=${TRAIN_MEM:-80}

# Run ID file on shared filesystem (accessible from compute nodes)
RUN_ID_FILE="$PROJECT_DIR/.run_id_${MODEL}_$$"

echo "=================================="
echo "Full Pipeline: $MODEL"
echo "=================================="
echo ""

# Stage 1: Extract (creates run_id)
echo "Submitting: extract stage"
EXTRACT_JOB=$(sbatch --parsable <<SLURM
#!/bin/bash
#SBATCH --job-name=caa_${MODEL}_extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=$PROJECT_DIR/logs/caa_${MODEL}_extract_%j.out
#SBATCH --error=$PROJECT_DIR/logs/caa_${MODEL}_extract_%j.err

export HF_HOME="$SHARE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SHARE_DIR/hf_cache/transformers"
export HF_DATASETS_CACHE="$SHARE_DIR/hf_cache/datasets"

cd $PROJECT_DIR
source venv/bin/activate

# Run and capture RUN_ID
output=\$(python -m src.stages.extract --model $MODEL --verbose 2>&1 | tee /dev/stderr)
run_id=\$(echo "\$output" | grep "^RUN_ID=" | cut -d= -f2)

if [ -n "\$run_id" ]; then
    echo "\$run_id" > $RUN_ID_FILE
    echo "Run ID saved: \$run_id"
fi
SLURM
)
echo "  Extract job: $EXTRACT_JOB"

# Stage 2: Train (reads run_id from file, or uses job output)
echo "Submitting: train stage"
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$EXTRACT_JOB <<SLURM
#!/bin/bash
#SBATCH --job-name=caa_${MODEL}_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:$TRAIN_GPUS
#SBATCH --cpus-per-task=$((TRAIN_GPUS * 10))
#SBATCH --mem=${TRAIN_MEM}G
#SBATCH --time=06:00:00
#SBATCH --output=$PROJECT_DIR/logs/caa_${MODEL}_train_%j.out
#SBATCH --error=$PROJECT_DIR/logs/caa_${MODEL}_train_%j.err

export HF_HOME="$SHARE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SHARE_DIR/hf_cache/transformers"
export HF_DATASETS_CACHE="$SHARE_DIR/hf_cache/datasets"

cd $PROJECT_DIR
source venv/bin/activate

RUN_ID=\$(cat $RUN_ID_FILE 2>/dev/null || echo "")
if [ -z "\$RUN_ID" ]; then
    echo "ERROR: No RUN_ID found. Extract stage may have failed."
    exit 1
fi

echo "Using RUN_ID: \$RUN_ID"
python -m src.stages.train --model $MODEL --run-id \$RUN_ID --verbose
SLURM
)
echo "  Train job: $TRAIN_JOB"

# Stage 3: Generate
echo "Submitting: generate stage"
GEN_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB <<SLURM
#!/bin/bash
#SBATCH --job-name=caa_${MODEL}_generate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_DIR/logs/caa_${MODEL}_generate_%j.out
#SBATCH --error=$PROJECT_DIR/logs/caa_${MODEL}_generate_%j.err

export HF_HOME="$SHARE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SHARE_DIR/hf_cache/transformers"
export HF_DATASETS_CACHE="$SHARE_DIR/hf_cache/datasets"

cd $PROJECT_DIR
source venv/bin/activate

RUN_ID=\$(cat $RUN_ID_FILE 2>/dev/null || echo "")
if [ -z "\$RUN_ID" ]; then
    echo "ERROR: No RUN_ID found."
    exit 1
fi

echo "Using RUN_ID: \$RUN_ID"
python -m src.stages.generate --model $MODEL --run-id \$RUN_ID --verbose
SLURM
)
echo "  Generate job: $GEN_JOB"

# Stage 4: Score (12B judge needs ~24GB VRAM)
echo "Submitting: score stage"
SCORE_JOB=$(sbatch --parsable --dependency=afterok:$GEN_JOB <<SLURM
#!/bin/bash
#SBATCH --job-name=caa_${MODEL}_score
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=$PROJECT_DIR/logs/caa_${MODEL}_score_%j.out
#SBATCH --error=$PROJECT_DIR/logs/caa_${MODEL}_score_%j.err

export HF_HOME="$SHARE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SHARE_DIR/hf_cache/transformers"
export HF_DATASETS_CACHE="$SHARE_DIR/hf_cache/datasets"

cd $PROJECT_DIR
source venv/bin/activate

RUN_ID=\$(cat $RUN_ID_FILE 2>/dev/null || echo "")
if [ -z "\$RUN_ID" ]; then
    echo "ERROR: No RUN_ID found."
    exit 1
fi

echo "Using RUN_ID: \$RUN_ID"
python -m src.stages.score --model $MODEL --run-id \$RUN_ID --verbose

# Cleanup
rm -f $RUN_ID_FILE
SLURM
)
echo "  Score job: $SCORE_JOB"

echo ""
echo "=================================="
echo "Pipeline submitted!"
echo ""
echo "Job chain: $EXTRACT_JOB -> $TRAIN_JOB -> $GEN_JOB -> $SCORE_JOB"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs:    tail -f $PROJECT_DIR/logs/caa_${MODEL}_*"
echo "=================================="
