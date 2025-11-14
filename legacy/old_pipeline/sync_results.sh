#!/bin/bash

# Sync results from remote SLURM server
# Usage: ./sync_results.sh username@server /path/to/project [--logs]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ $# -lt 2 ]; then
    echo "Usage: $0 username@server /path/to/remote/project [--logs]"
    echo "  --logs  Also sync log files"
    exit 1
fi

SERVER=$1
REMOTE_PATH=$2
SYNC_LOGS=false

if [ "$3" == "--logs" ]; then
    SYNC_LOGS=true
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Syncing Results from Remote${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Sync results
echo -e "${YELLOW}Syncing results...${NC}"
mkdir -p results
rsync -avz --progress ${SERVER}:${REMOTE_PATH}/results/ ./results/

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Results synced successfully${NC}"
else
    echo -e "${RED}Warning: Some results may have failed to sync${NC}"
fi

# Sync logs if requested
if [ "$SYNC_LOGS" = true ]; then
    echo ""
    echo -e "${YELLOW}Syncing logs...${NC}"
    mkdir -p logs
    rsync -avz --progress ${SERVER}:${REMOTE_PATH}/logs/ ./logs/

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Logs synced successfully${NC}"
    fi
fi

# Show summary
echo ""
echo -e "${YELLOW}=== Results Summary ===${NC}"

# Count result files
if [ -d "results" ]; then
    echo "Result directories:"
    find results -maxdepth 1 -type d | tail -n +2 | while read dir; do
        count=$(find "$dir" -name "*.json" 2>/dev/null | wc -l)
        echo "  $(basename $dir): $count JSON files"
    done

    echo ""
    echo "Recent results:"
    find results -name "*.json" -printf "%T+ %p\n" 2>/dev/null | sort -r | head -5 | while read line; do
        echo "  $line"
    done
fi

if [ "$SYNC_LOGS" = true ] && [ -d "logs" ]; then
    echo ""
    echo "Log files:"
    ls -lht logs/*.out 2>/dev/null | head -5
fi

echo ""
echo -e "${GREEN}Sync complete!${NC}"
echo ""
echo "To analyze results:"
echo "  python analyze_results.py"