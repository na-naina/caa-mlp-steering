#!/bin/bash
#SBATCH --job-name=clean_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=00:15:00
#SBATCH --partition=compute
#SBATCH --output=logs/clean_%j.out
#SBATCH --error=logs/clean_%j.err

echo "=================================================="
echo "Cleaning Corrupted Cache in Shared Storage"
echo "=================================================="
echo ""

SHARED_DIR="/springbrook/share/dcsresearch/u5584851"
PROJECT_DIR="$SHARED_DIR/caa_experiments"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Project directory doesn't exist yet"
    exit 0
fi

cd "$PROJECT_DIR"

echo "Current cache status:"
du -sh cache/* 2>/dev/null || echo "No cache yet"
echo ""

echo "Looking for corrupted files..."

# Find and remove corrupted safetensor files
find cache -name "*.safetensors" -type f 2>/dev/null | while read file; do
    # Check if file is empty or very small (likely corrupted)
    size=$(stat -c%s "$file" 2>/dev/null || echo 0)
    if [ "$size" -lt 1000 ]; then
        echo "  Removing small/empty file: $file (${size} bytes)"
        rm -f "$file"
    fi
done

# Remove incomplete downloads
find cache -name "*.lock" -type f -exec rm -v {} \; 2>/dev/null
find cache -name "*.tmp*" -type f -exec rm -v {} \; 2>/dev/null
find cache -name "*.part" -type f -exec rm -v {} \; 2>/dev/null

# Remove gemma-2-2b cache specifically (it's corrupted)
echo ""
echo "Removing gemma-2-2b cache to force re-download..."
rm -rf cache/huggingface/hub/models--google--gemma-2-2b*
rm -rf cache/transformers/models--google--gemma-2-2b*

echo ""
echo "Cache cleaned. Status after cleaning:"
du -sh cache/* 2>/dev/null || echo "Cache is empty"

echo ""
echo "=================================================="
echo "Cache Cleanup Complete"
echo "=================================================="