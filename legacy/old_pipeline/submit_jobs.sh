#!/bin/bash
# Job submission helper script for CAA experiments

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup        - Set up Python environment and install dependencies"
    echo "  test         - Run a quick test with gemma2-2b"
    echo "  small        - Submit job for small models (2B)"
    echo "  medium       - Submit job for medium models (7B-9B)"  
    echo "  large        - Submit job for large models (27B)"
    echo "  array        - Submit array job for all models"
    echo "  status       - Check status of running jobs"
    echo "  analyze      - Analyze completed results"
    echo "  clean        - Clean up temporary files"
    echo ""
    echo "Options:"
    echo "  --model MODEL     - Specify model (e.g., gemma2-2b)"
    echo "  --eval MODE       - Evaluation mode: quick|standard|full"
    echo "  --scale GRAN      - Scale granularity: fine|standard|coarse"
    echo "  --dry-run         - Show what would be submitted without submitting"
}

function setup_environment() {
    echo -e "${GREEN}Setting up Python environment...${NC}"
    
    # Load modules
    module purge
    module load Python/3.11.5-GCCcore-13.2.0
    module load CUDA/12.2.0
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python -m venv venv
    fi
    
    source venv/bin/activate
    
    # Install requirements
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo -e "${GREEN}Environment setup complete!${NC}"
}

function run_test() {
    echo -e "${YELLOW}Running quick test...${NC}"
    
    source venv/bin/activate
    
    python caa_experiment.py \
        --model_name google/gemma-2b \
        --layer 9 \
        --scales 0 1.0 5.0 \
        --num_samples 10 \
        --output_dir test_results
    
    echo -e "${GREEN}Test complete! Check test_results/ for output${NC}"
}

function submit_job() {
    local job_type=$1
    local model=$2
    local eval_mode=$3
    local scale_gran=$4
    local dry_run=$5
    
    case $job_type in
        small)
            script="run_small_model.slurm"
            ;;
        medium|large)
            script="run_large_model.slurm"
            ;;
        array)
            script="run_array.slurm"
            ;;
        *)
            echo -e "${RED}Unknown job type: $job_type${NC}"
            exit 1
            ;;
    esac
    
    # Build sbatch command
    cmd="sbatch"
    
    if [ ! -z "$model" ]; then
        cmd="$cmd --export=MODEL=$model"
    fi
    if [ ! -z "$eval_mode" ]; then
        cmd="$cmd,EVAL_MODE=$eval_mode"
    fi
    if [ ! -z "$scale_gran" ]; then
        cmd="$cmd,SCALE_GRAN=$scale_gran"
    fi
    
    cmd="$cmd $script"
    
    if [ "$dry_run" = true ]; then
        echo -e "${YELLOW}Dry run - would execute:${NC}"
        echo "$cmd"
    else
        echo -e "${GREEN}Submitting job...${NC}"
        echo "Command: $cmd"
        $cmd
    fi
}

function check_status() {
    echo -e "${GREEN}Current SLURM jobs:${NC}"
    squeue -u $USER --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
    
    echo -e "\n${GREEN}Recent completed jobs:${NC}"
    sacct -u $USER --format="JobID%20,JobName%30,State,ExitCode,Elapsed" -S $(date -d '1 day ago' '+%Y-%m-%d')
}

function analyze_results() {
    local results_dir=$1
    
    if [ -z "$results_dir" ]; then
        # Find most recent results directory
        results_dir=$(ls -dt results/slurm_* results/array_* results/batch_run_* 2>/dev/null | head -n1)
        
        if [ -z "$results_dir" ]; then
            echo -e "${RED}No results directory found${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}Analyzing results in: $results_dir${NC}"
    
    source venv/bin/activate
    python analyze_results.py --results_dir "$results_dir" --output_dir "analysis/$(basename $results_dir)"
}

function clean_temp() {
    echo -e "${YELLOW}Cleaning temporary files...${NC}"
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -type f -name "*.pyc" -delete
    
    # Clean old slurm logs (keep last 10)
    if [ -d "slurm_logs" ]; then
        ls -t slurm_logs/*.out 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null
        ls -t slurm_logs/*.err 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null
    fi
    
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Parse arguments
COMMAND=$1
shift

MODEL=""
EVAL_MODE=""
SCALE_GRAN=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --eval)
            EVAL_MODE="$2"
            shift 2
            ;;
        --scale)
            SCALE_GRAN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Execute command
case $COMMAND in
    setup)
        setup_environment
        ;;
    test)
        run_test
        ;;
    small)
        submit_job small "$MODEL" "$EVAL_MODE" "$SCALE_GRAN" $DRY_RUN
        ;;
    medium)
        submit_job medium "$MODEL" "$EVAL_MODE" "$SCALE_GRAN" $DRY_RUN
        ;;
    large)
        submit_job large "$MODEL" "$EVAL_MODE" "$SCALE_GRAN" $DRY_RUN
        ;;
    array)
        submit_job array "" "$EVAL_MODE" "$SCALE_GRAN" $DRY_RUN
        ;;
    status)
        check_status
        ;;
    analyze)
        analyze_results "$2"
        ;;
    clean)
        clean_temp
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        print_usage
        exit 1
        ;;
esac
