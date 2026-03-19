#!/usr/bin/env python3
"""Evaluate LoRA DPO model WITHOUT steering (baseline for combination experiment).

Usage:
    python scripts/eval_lora_only.py --lora-path data/outputs/lora_dpo/lora_adapter
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.truthfulqa import TruthfulQADatasetManager
from src.evaluation.truthfulqa import evaluate_multiple_choice, evaluate_generation

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/lora_dpo_only"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    LOG.info("Loading LoRA model")
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(args.lora_path))
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    dataset = TruthfulQADatasetManager(seed=args.seed)
    splits = dataset.create_pipeline_splits(steering_pool_size=100, train_size=309, test_size=0)

    test_items = dataset.get_items(splits.test)
    mc_indices = [i for i in splits.test if dataset.is_valid_mc(i)]
    mc_items = dataset.get_items(mc_indices)

    # MC eval (no steering)
    mc_result = evaluate_multiple_choice(
        model, tokenizer, mc_items,
        layer_index=8, steering_vector=None, scale=0.0,
        max_length=512, primary_device=device,
    )
    LOG.info("MC accuracy (LoRA only, no steering): %.4f", mc_result["stats"].accuracy)

    # Generate (no steering)
    gen_cfg = {
        "preset": "qa", "temperature": 0.3, "top_p": 0.9,
        "max_new_tokens": 64, "max_length": 512,
        "stop_sequences": ["\n\n", "\nQuestion:"],
    }
    gen_result = evaluate_generation(
        model, tokenizer, test_items,
        layer_index=8, steering_vector=None, scale=0.0,
        generation_cfg=gen_cfg, primary_device=device,
        judge=None, semantic_judge=None,
    )

    out_dir = args.output_dir / "mlp_mc" / "scale_1.00"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "generation_details.json").open("w") as f:
        json.dump(gen_result["details"], f, indent=2)

    LOG.info("Done! %d responses saved to %s", len(gen_result["details"]), out_dir)


if __name__ == "__main__":
    main()
