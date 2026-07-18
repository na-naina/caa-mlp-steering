#!/bin/bash
# Gemma3-12B layer sweep: train-only at each layer, check MC accuracy
# No gradient checkpointing (breaks Gemma3 training)
set -e

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

LAYERS="8 12 16 20 24 28 32 36 40"
OUTDIR="data/outputs/gemma3_sweep"
mkdir -p "$OUTDIR"

for layer in $LAYERS; do
    echo "========================================"
    echo "LAYER $layer — $(date)"
    echo "========================================"
    python3 run.py \
        --model gemma3_12b_L${layer} \
        --stage train-only \
        --output-dir "${OUTDIR}/L${layer}" \
        2>&1 | tee "${OUTDIR}/L${layer}.log"
    echo "LAYER $layer DONE — $(date)"
    echo ""
done

echo "========================================"
echo "SWEEP COMPLETE — Summary of MC accuracies:"
echo "========================================"
for layer in $LAYERS; do
    echo -n "Layer $layer: "
    grep -o "mc_accuracy.*" "${OUTDIR}/L${layer}.log" | tail -1 || echo "NOT FOUND"
done
