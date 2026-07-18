#!/usr/bin/env python3
"""Causal test: is a fine-tuned model's truthfulness gain mediated by one direction?

Arditi-style directional ablation. Generates TruthfulQA test answers under:
  base            frozen base model
  dpo             base + LoRA-DPO adapter
  dpo_ablated     base + LoRA-DPO, with the v_MAST direction projected OUT of the
                  residual stream (h <- h - (h.v_hat) v_hat) during generation
  base_ablated    control: ablation on the base model (should be ~neutral)

If dpo_ablated collapses toward base, the adapter's truthfulness gain is carried
by (roughly) that single direction -> "truthfulness fine-tuning is mostly one
direction". If it survives, fine-tuning uses capacity beyond the direction.

Judging: run scripts/evaluate_with_gpt_judge.py on the output dir afterwards.

Usage:
    python scripts/eval_direction_ablation.py \
        --run-dir data/outputs/multiseed/seed_42 \
        --lora-path <path-to-lora-adapter> \
        --layers 8            # or e.g. --layers all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import load_causal_model
from src.data.truthfulqa import TruthfulQADatasetManager
from src.evaluation.truthfulqa import evaluate_generation
from src.steering.apply import _get_decoder_layer

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


def make_ablation_hook(direction: torch.Tensor):
    def hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d = direction.to(hidden.device, dtype=hidden.dtype)
        d = d / d.norm()
        coeff = (hidden * d).sum(dim=-1, keepdim=True)
        hidden = hidden - coeff * d
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run dir providing vectors/v_mast_mc.pt, splits, config")
    p.add_argument("--lora-path", type=Path, required=True)
    p.add_argument("--vector-file", default="vectors/v_mast_mc.pt")
    p.add_argument("--layers", default="8",
                   help="Comma-separated layer indices for ablation, or 'all'")
    p.add_argument("--variants", nargs="+",
                   default=["base", "dpo", "dpo_ablated", "base_ablated"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    model_name = cfg["model"]["name"]

    dataset = TruthfulQADatasetManager(seed=args.seed)
    splits = dataset.create_pipeline_splits(steering_pool_size=100, train_size=309, test_size=0)
    test_items = dataset.get_items(splits.test)

    direction = torch.load(args.run_dir / args.vector_file, map_location="cpu").float()
    gen_cfg = {"preset": "qa", "temperature": 0.3, "top_p": 0.9, "max_new_tokens": 64,
               "max_length": 512, "stop_sequences": ["\n\n", "\nQuestion:"]}

    def build(with_lora: bool):
        loaded = load_causal_model(model_name, dtype="bfloat16", device_map="auto")
        model, tok, dev = loaded.model, loaded.tokenizer, loaded.primary_device
        if with_lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(args.lora_path))
            model = model.merge_and_unload()
        model.eval()
        return model, tok, dev

    def layer_indices(model):
        if args.layers == "all":
            base = model
            n = len(_get_decoder_layer(base, 0).__self__ ) if False else None
            # fall back: probe config
            return list(range(model.config.num_hidden_layers))
        return [int(x) for x in args.layers.split(",")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        with_lora = variant.startswith("dpo")
        ablate = variant.endswith("_ablated")
        LOG.info("=== %s (lora=%s, ablate=%s) ===", variant, with_lora, ablate)
        model, tok, dev = build(with_lora)
        handles = []
        if ablate:
            for li in layer_indices(model):
                handles.append(_get_decoder_layer(model, li)
                               .register_forward_hook(make_ablation_hook(direction)))
        result = evaluate_generation(model, tok, test_items, layer_index=cfg["model"]["layer"],
                                     steering_vector=None, scale=0.0,
                                     generation_cfg=gen_cfg, primary_device=dev,
                                     judge=None, semantic_judge=None)
        for h in handles:
            h.remove()
        vdir = args.output_dir / variant / "scale_0.00"
        vdir.mkdir(parents=True, exist_ok=True)
        with (vdir / "generation_details.json").open("w") as f:
            json.dump(result["details"], f, indent=2)
        LOG.info("%s: %d generations saved", variant, len(result["details"]))
        del model
        torch.cuda.empty_cache()

    LOG.info("Done. Judge with: python scripts/evaluate_with_gpt_judge.py evaluate --model %s",
             args.output_dir.name)


if __name__ == "__main__":
    main()
