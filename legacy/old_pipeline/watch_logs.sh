#!/bin/bash

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CAA Experiment Log Monitor${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Function to check job status
check_jobs() {
    echo -e "${YELLOW}Current Jobs:${NC}"
    squeue -u $USER --format="%.10i %.20j %.8u %.2t %.10M %.6D %R"
    echo ""
}

# Function to show recent logs
show_recent_logs() {
    echo -e "${YELLOW}Recent Log Files:${NC}"
    ls -lht logs/*.out 2>/dev/null | head -5
    echo ""
}

# Function to check for errors
check_errors() {
    echo -e "${YELLOW}Recent Errors:${NC}"
    for err_file in $(ls -t logs/*.err 2>/dev/null | head -2); do
        if [ -s "$err_file" ]; then
            echo -e "${RED}$(basename $err_file):${NC}"
            tail -5 "$err_file"
            echo ""
        fi
    done
}

# Function to show progress from logs
show_progress() {
    echo -e "${YELLOW}Experiment Progress:${NC}"

    # Check for CAA extraction
    echo -n "CAA Extraction: "
    grep -l "Extracting CAA vectors" logs/*.out 2>/dev/null | wc -l

    # Check for MLP training
    echo -n "MLP Training: "
    grep -l "Training MLP processor" logs/*.out 2>/dev/null | wc -l

    # Check for evaluation
    echo -n "Evaluations: "
    grep -l "Evaluating at scale" logs/*.out 2>/dev/null | wc -l

    # Check for completions
    echo -n "Completed: "
    grep -l "Results saved to" logs/*.out 2>/dev/null | wc -l

    echo ""
}

# Main monitoring loop
if [ "$1" == "-w" ] || [ "$1" == "--watch" ]; then
    # Continuous monitoring mode
    while true; do
        clear
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}CAA Experiment Monitor ($(date '+%H:%M:%S'))${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""

        check_jobs
        show_recent_logs
        check_errors
        show_progress

        echo ""
        echo "Press Ctrl+C to exit"
        sleep 5
    done
else
    # Single check
    check_jobs
    show_recent_logs
    check_errors
    show_progress

    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  ./watch_logs.sh -w        # Watch continuously"
    echo "  tail -f logs/*.out        # Follow all output logs"
    echo "  tail -f logs/*.err        # Follow all error logs"
fi
