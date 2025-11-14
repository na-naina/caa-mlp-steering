#!/bin/bash

# --- CONFIGURATION ---
# This is the SHARED cache directory your SLURM jobs actually use.
SHARED_CACHE_DIR="/springbrook/share/dcsresearch/u5584851/caa_experiments/cache"
# ---------------------

echo "=================================================="
echo "  Deep Clean SHARED Model Cache"
echo "  Target: ${SHARED_CACHE_DIR}"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}WARNING: This will remove the SHARED model cache at the path specified above.${NC}"
echo "This will force all jobs to re-download models."
echo "Continue? (yes/no)"
read -r response

if [[ "$response" != "yes" ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo -e "${YELLOW}Step 1: Cancelling your running jobs to prevent conflicts${NC}"
scancel -u "$USER"
echo "Waiting a moment for jobs to terminate..."
sleep 3

echo ""
echo -e "${YELLOW}Step 2: Removing shared cache directory${NC}"

if [ -d "$SHARED_CACHE_DIR" ]; then
    echo "  Removing: $SHARED_CACHE_DIR"
    rm -rf "$SHARED_CACHE_DIR"
    echo -e "  ${GREEN}✓ Directory removed.${NC}"
else
    echo -e "  ${GREEN}✓ Directory does not exist, nothing to remove.${NC}"
fi

echo ""
echo -e "${YELLOW}Step 3: Creating fresh shared cache directories${NC}"
mkdir -p "${SHARED_CACHE_DIR}/huggingface"
mkdir -p "${SHARED_CACHE_DIR}/transformers"
mkdir -p "${SHARED_CACHE_DIR}/datasets"
echo -e "  ${GREEN}✓ Fresh directories created.${NC}"

# Copy token to new shared cache if it exists in the home directory
if [ -f ~/.cache/huggingface/token ]; then
    mkdir -p "${SHARED_CACHE_DIR}/huggingface"
    cp ~/.cache/huggingface/token "${SHARED_CACHE_DIR}/huggingface/token"
    echo -e "  ${GREEN}✓ Token copied to new shared cache.${NC}"
fi

echo ""
echo -e "${YELLOW}Step 4: Checking disk space on the shared volume${NC}"
df -h "$SHARED_CACHE_DIR" | head -2

echo ""
echo "=================================================="
echo -e "${GREEN}Shared Cache Cleanup Complete!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Submit ONE small job first to populate the cache:"
echo "   python submit_core_experiments.py --filter gemma2-2b"
echo ""
echo "2. Once it runs successfully, submit your other jobs."
