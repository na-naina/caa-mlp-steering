#!/usr/bin/env python3
"""Generate baseline (unsteered) responses for any model.

Usage:
    python scripts/generate_baseline.py --model google/gemma-3-12b-it --output-dir data/outputs/gemma3_12b_baseline
    python scripts/generate_baseline.py --model meta-llama/Llama-2-7b-chat-hf --output-dir data/outputs/llama2_baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import load_causal_model
from src.data.truthfulqa import TruthfulQADatasetManager
from src.evaluation.truthfulqa import evaluate_generation

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loaded = load_causal_model(args.model, dtype="bfloat16", device_map="auto")
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.primary_device
    model.eval()

    dataset = TruthfulQADatasetManager(seed=args.seed)
    splits = dataset.create_pipeline_splits(steering_pool_size=100, train_size=309, test_size=0)
    test_items = dataset.get_items(splits.test)

    gen_cfg = {
        "preset": "qa", "temperature": 0.3, "top_p": 0.9,
        "max_new_tokens": 64, "max_length": 512,
        "stop_sequences": ["\n\n", "\nQuestion:"],
    }

    LOG.info("Generating baseline (no steering) for %s", args.model)
    result = evaluate_generation(
        model, tokenizer, test_items,
        layer_index=args.layer, steering_vector=None, scale=0.0,
        generation_cfg=gen_cfg, primary_device=device,
        judge=None, semantic_judge=None,
    )

    out = args.output_dir / "mlp_mc" / "scale_1.00"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "generation_details.json").open("w") as f:
        json.dump(result["details"], f, indent=2)

    LOG.info("Done! %d responses saved to %s", len(result["details"]), out)


if __name__ == "__main__":
    main()
