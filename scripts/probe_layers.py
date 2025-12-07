#!/usr/bin/env python3
"""Probe multiple layers to find which differentiate pos/neg activations.

This helps debug zero-norm CAA vector issues by testing all layers
without running the full pipeline.

Usage:
    python scripts/probe_layers.py --model google/gemma-3-12b-it
    python scripts/probe_layers.py --model google/gemma-3-12b-it --layers 10 15 20 25 30
    python scripts/probe_layers.py --model google/gemma-3-12b-it --num-samples 20

Output:
    Table showing diff_norm for each layer - higher is better.
    Layers with diff_norm < 1e-6 will produce zero-norm CAA vectors.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


def get_num_layers(model) -> int:
    """Get total number of decoder layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        if hasattr(model.model.language_model, "layers"):
            return len(model.model.language_model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError("Cannot determine number of layers")


def get_layer(model, idx: int):
    """Get specific decoder layer."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[idx]
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        if hasattr(model.model.language_model, "layers"):
            return model.model.language_model.layers[idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[idx]
    raise ValueError("Cannot get layer")


def extract_at_layer(model, tokenizer, texts: List[str], layer_idx: int, device) -> torch.Tensor:
    """Extract mean activations at a specific layer."""
    layer = get_layer(model, layer_idx)
    activations = []

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activations.append(hidden.detach().float())

    handle = layer.register_forward_hook(hook)
    try:
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        attention_mask = encoded.get("attention_mask")

        with torch.no_grad():
            model(**encoded)

        hidden = activations[0]
        # Mean over non-padding tokens
        mask = attention_mask.to(hidden.device).unsqueeze(-1).float()
        token_counts = mask.sum(dim=1).clamp(min=1.0)
        mean_hidden = (hidden * mask).sum(dim=1) / token_counts

        return mean_hidden.cpu()
    finally:
        handle.remove()


def build_sample_prompts(num_samples: int = 50):
    """Build contrastive prompt pairs from TruthfulQA."""
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "generation", split="validation")

    pos_prompts = []
    neg_prompts = []

    for i, item in enumerate(ds):
        if i >= num_samples:
            break

        question = item["question"]
        correct = item.get("correct_answers", [])
        incorrect = item.get("incorrect_answers", [])

        if correct and incorrect:
            pos_prompts.append(f"Q: {question}\nA: {correct[0]}")
            neg_prompts.append(f"Q: {question}\nA: {incorrect[0]}")

    return pos_prompts, neg_prompts


def probe_layers(
    model_name: str,
    layers: Optional[List[int]] = None,
    num_samples: int = 50,
    dtype: str = "bfloat16",
):
    """Probe all/specified layers and report diff norms."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Setup
    torch_dtype = getattr(torch, dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOG.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = get_num_layers(model)
    LOG.info("Model has %d layers", num_layers)

    # Determine which layers to probe
    if layers is None:
        # Probe every 4th layer + first and last
        layers = [0] + list(range(4, num_layers, 4)) + [num_layers - 1]
        layers = sorted(set(layers))

    LOG.info("Probing layers: %s", layers)

    # Build prompts
    LOG.info("Building %d sample prompt pairs", num_samples)
    pos_prompts, neg_prompts = build_sample_prompts(num_samples)
    LOG.info("Got %d valid pairs", len(pos_prompts))

    if not pos_prompts:
        LOG.error("No valid prompt pairs found")
        return

    # Probe each layer
    results = []
    primary_device = next(model.parameters()).device

    for layer_idx in layers:
        LOG.info("Probing layer %d/%d...", layer_idx, num_layers - 1)

        try:
            pos_acts = extract_at_layer(model, tokenizer, pos_prompts, layer_idx, primary_device)
            neg_acts = extract_at_layer(model, tokenizer, neg_prompts, layer_idx, primary_device)

            pos_mean = pos_acts.mean(dim=0)
            neg_mean = neg_acts.mean(dim=0)
            diff = pos_mean - neg_mean
            diff_norm = diff.norm().item()

            pos_norm = pos_mean.norm().item()
            neg_norm = neg_mean.norm().item()

            results.append({
                "layer": layer_idx,
                "diff_norm": diff_norm,
                "pos_norm": pos_norm,
                "neg_norm": neg_norm,
                "status": "OK" if diff_norm > 1e-3 else ("WARN" if diff_norm > 1e-6 else "ZERO"),
            })

        except Exception as e:
            LOG.error("Failed at layer %d: %s", layer_idx, e)
            results.append({
                "layer": layer_idx,
                "diff_norm": None,
                "pos_norm": None,
                "neg_norm": None,
                "status": "ERROR",
            })

    # Print results
    print("\n" + "=" * 70)
    print(f"Layer Probe Results: {model_name}")
    print("=" * 70)
    print(f"{'Layer':>6} {'Diff Norm':>12} {'Pos Norm':>12} {'Neg Norm':>12} {'Status':>8}")
    print("-" * 70)

    best_layer = None
    best_norm = 0

    for r in results:
        if r["diff_norm"] is not None:
            print(f"{r['layer']:>6} {r['diff_norm']:>12.4e} {r['pos_norm']:>12.4e} {r['neg_norm']:>12.4e} {r['status']:>8}")
            if r["diff_norm"] > best_norm:
                best_norm = r["diff_norm"]
                best_layer = r["layer"]
        else:
            print(f"{r['layer']:>6} {'N/A':>12} {'N/A':>12} {'N/A':>12} {r['status']:>8}")

    print("-" * 70)

    if best_layer is not None:
        print(f"\nBest layer: {best_layer} (diff_norm = {best_norm:.4e})")

        # Recommend based on results
        good_layers = [r["layer"] for r in results if r["diff_norm"] and r["diff_norm"] > 1e-2]
        if good_layers:
            print(f"Recommended layers (diff_norm > 1e-2): {good_layers}")

        zero_layers = [r["layer"] for r in results if r["status"] == "ZERO"]
        if zero_layers:
            print(f"Avoid these layers (zero-norm): {zero_layers}")
    else:
        print("\nWARNING: No layers with usable diff_norm found!")
        print("This model may not be suitable for CAA steering with these prompts.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Probe layers for CAA vector quality")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--layers", type=int, nargs="+", help="Specific layers to probe")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of prompt pairs")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hf-cache", type=str, help="HuggingFace cache directory")
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    probe_layers(
        model_name=args.model,
        layers=args.layers,
        num_samples=args.num_samples,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
