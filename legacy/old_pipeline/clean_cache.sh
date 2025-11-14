#!/bin/bash

echo "=================================================="
echo "Cleaning Corrupted Model Cache"
echo "=================================================="
echo ""

# Find and remove corrupted JSON files
echo "Looking for model cache directories..."

# Check local cache
if [ -d ".cache" ]; then
    echo "Found local cache at .cache/"

    # Find model.safetensors.index.json files
    find .cache -name "*.json" -type f 2>/dev/null | while read json_file; do
        # Check if JSON file is valid
        python -m json.tool "$json_file" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "  ✗ Corrupted: $json_file"
            echo "    Removing..."
            rm -f "$json_file"

            # Also remove associated safetensors files in same directory
            dir=$(dirname "$json_file")
            echo "    Cleaning directory: $dir"
            rm -rf "$dir"
        fi
    done

    echo ""
    echo "Checking for incomplete downloads..."
    find .cache -name "*.lock" -type f -exec rm -v {} \;
    find .cache -name "*.tmp*" -type f -exec rm -v {} \;
fi

# Clean specific model caches if they exist
for model in "gemma-2-2b" "gemma-2-9b" "gemma-2-27b"; do
    cache_dir=".cache/huggingface/hub/models--google--${model}"
    if [ -d "$cache_dir" ]; then
        echo ""
        echo "Found cache for $model"
        echo "Remove it? (y/n)"
        read -r response
        if [[ "$response" == "y" ]]; then
            rm -rf "$cache_dir"
            echo "  ✓ Removed $model cache"
        fi
    fi
done

echo ""
echo "=================================================="
echo "Cache Cleanup Complete"
echo "=================================================="
echo ""
echo "You can now resubmit the jobs. The models will be re-downloaded."