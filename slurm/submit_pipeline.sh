#!/bin/bash
# CAA Steering Pipeline Submission
# Submits all stages as separate jobs with dependencies
#
# Usage:
#   ./submit_pipeline.sh <model_name>              # Full pipeline
#   ./submit_pipeline.sh <model_name> --from train # Start from train stage
#   ./submit_pipeline.sh <model_name> --only extract # Single stage
#
# Each stage requests appropriate resources:
#   extract:  1 GPU, inference only (~24GB)
#   train:    2 GPUs, needs gradients (~44GB)
#   generate: 1 GPU, inference only (~24GB)
#   score:    1 GPU, judge model (~24GB)

set -e

MODEL=${1:?"Usage: ./submit_pipeline.sh <model_name> [--from stage] [--only stage] [--run-id ID]"}
shift

# Parse options
START_STAGE="extract"
ONLY_STAGE=""
RUN_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --from)
            START_STAGE="$2"
            shift 2
            ;;
        --only)
            ONLY_STAGE="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Paths
PROJECT_DIR="/springbrook/share/dcsresearch/u5584851/caa_steering"
SHARE_DIR="/springbrook/share/dcsresearch/u5584851"
CONFIG_FILE="$PROJECT_DIR/configs/models/$MODEL.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config not found: $CONFIG_FILE"
    exit 1
fi

# Ensure logs directory exists
mkdir -p "$PROJECT_DIR/logs"

# Extract SLURM settings from config (for large model overrides)
get_config_value() {
    grep -A10 "^slurm:" "$CONFIG_FILE" | grep "$1:" | awk '{print $2}' | tr -d '"' || echo ""
}

BASE_GPUS=$(get_config_value "gpus")
BASE_MEM=$(get_config_value "mem_gb")
BASE_TIME=$(get_config_value "time")

# Stage-specific resource configurations
# Format: GPUS MEM_GB TIME
declare -A STAGE_RESOURCES
# Resources: GPUS MEM_GB TIME
# 12B judge for scoring needs ~24GB VRAM (inference only)
STAGE_RESOURCES[extract]="1 40 02:00:00"
STAGE_RESOURCES[train]="${BASE_GPUS:-2} ${BASE_MEM:-80} 06:00:00"
STAGE_RESOURCES[generate]="1 40 04:00:00"
STAGE_RESOURCES[score]="1 40 03:00:00"

# Order of stages
STAGES=("extract" "train" "generate" "score")

# Function to submit a stage
submit_stage() {
    local stage=$1
    local dependency=$2
    local run_id_arg=$3

    read -r gpus mem time <<< "${STAGE_RESOURCES[$stage]}"
    local cpus=$((gpus * 10))
    local job_name="caa_${MODEL}_${stage}"

    local dep_flag=""
    if [ -n "$dependency" ]; then
        dep_flag="#SBATCH --dependency=afterok:$dependency"
    fi

    local run_id_opt=""
    if [ -n "$run_id_arg" ]; then
        run_id_opt="--run-id $run_id_arg"
    fi

    echo "Submitting: $job_name (GPUs=$gpus, Mem=${mem}G, Time=$time)"

    job_id=$(sbatch --parsable <<SLURM
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:$gpus
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=${mem}G
#SBATCH --time=$time
#SBATCH --output=$PROJECT_DIR/logs/${job_name}_%j.out
#SBATCH --error=$PROJECT_DIR/logs/${job_name}_%j.err
$dep_flag

# Environment setup
export HF_HOME="$SHARE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SHARE_DIR/hf_cache/transformers"
export HF_DATASETS_CACHE="$SHARE_DIR/hf_cache/datasets"

echo "=== Stage: $stage ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$(hostname)"
echo "GPUs: \$CUDA_VISIBLE_DEVICES"
echo "Started: \$(date)"
echo ""

cd $PROJECT_DIR
source venv/bin/activate

# Run stage
python -m src.stages.$stage --model $MODEL $run_id_opt --verbose

echo ""
echo "Finished: \$(date)"
SLURM
)

    echo "  Job ID: $job_id"
    echo "$job_id"
}

echo "=================================="
echo "CAA Steering Pipeline: $MODEL"
echo "=================================="
echo ""

# Determine which stages to run
if [ -n "$ONLY_STAGE" ]; then
    STAGES=("$ONLY_STAGE")
    if [ -z "$RUN_ID" ] && [ "$ONLY_STAGE" != "extract" ]; then
        echo "Error: --run-id required for --only $ONLY_STAGE"
        exit 1
    fi
else
    # Find start index
    start_idx=0
    for i in "${!STAGES[@]}"; do
        if [ "${STAGES[$i]}" = "$START_STAGE" ]; then
            start_idx=$i
            break
        fi
    done
    STAGES=("${STAGES[@]:$start_idx}")

    if [ "$START_STAGE" != "extract" ] && [ -z "$RUN_ID" ]; then
        echo "Error: --run-id required when starting from $START_STAGE"
        exit 1
    fi
fi

echo "Stages to run: ${STAGES[*]}"
echo ""

# Submit stages with dependencies
prev_job=""
for stage in "${STAGES[@]}"; do
    if [ "$stage" = "extract" ] && [ -z "$RUN_ID" ]; then
        # Extract stage outputs RUN_ID, capture it
        job_id=$(submit_stage "$stage" "$prev_job" "")
        # Note: RUN_ID will be in the job output, subsequent jobs need it
        # For simplicity, we require --run-id for resumed pipelines
        echo ""
        echo "NOTE: After extract completes, find RUN_ID in job output and use:"
        echo "  ./submit_pipeline.sh $MODEL --from train --run-id <RUN_ID>"
        break
    else
        job_id=$(submit_stage "$stage" "$prev_job" "$RUN_ID")
    fi
    prev_job=$job_id
done

echo ""
echo "=================================="
echo "Jobs submitted. Monitor with: squeue -u \$USER"
echo "=================================="
