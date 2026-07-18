#!/usr/bin/env python3
"""Canonical MC1/MC2 evaluation via lm-evaluation-harness, with steering.

This is THE definition of MC1/MC2 for this project. It runs the harness's
`truthfulqa_mc1` / `truthfulqa_mc2` tasks (which include the standard 6-question
QA primer, summed log-prob scoring, no chat template) for the unsteered
baseline and for a saved steering vector, then reports both the full-817
metrics and the metrics filtered to a run's held-out test split.

Usage:
    python scripts/eval_mc_harness.py --run-dir data/outputs/multiseed/seed_42
    python scripts/eval_mc_harness.py --run-dir <dir> --variants steered
    python scripts/eval_mc_harness.py --run-dir <dir> --vector-file vectors/base_vector.pt --label raw_caa

Outputs <run-dir>/mc_harness_results.json with per-variant, per-task metrics
(full + test-split-filtered), harness version, and per-sample records.
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

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)

TASKS = ["truthfulqa_mc1", "truthfulqa_mc2"]


def make_steering_hook(vector: torch.Tensor, scale: float):
    def hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steer = scale * vector.to(hidden.device, dtype=hidden.dtype)
        hidden = hidden + steer.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


def get_decoder_layer(model, layer_index: int):
    from src.steering.apply import _get_decoder_layer
    return _get_decoder_layer(model, layer_index)


def load_test_questions(run_dir: Path) -> set[str]:
    """Map the run's test-split indices to question strings."""
    splits = json.loads((run_dir / "metadata" / "splits.json").read_text())
    test_indices = splits["test"] if isinstance(splits, dict) else splits
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation")["validation"]
    return {ds[int(i)]["question"].strip() for i in test_indices}


def sample_records(samples: list[dict]) -> list[dict]:
    """Extract (question, acc) per sample, tolerating lm-eval layout changes."""
    records = []
    for s in samples:
        doc = s.get("doc", {})
        question = (doc.get("question") or "").strip()
        acc = s.get("acc")
        if acc is None and isinstance(s.get("metrics"), dict):
            acc = s["metrics"].get("acc")
        if acc is None:
            for k, v in s.items():
                if k.startswith("acc"):
                    acc = v
                    break
        records.append({"question": question, "acc": float(acc)})
    return records


def aggregate(records: list[dict], keep: set[str] | None = None) -> dict:
    vals = [r["acc"] for r in records if keep is None or r["question"] in keep]
    return {"n": len(vals), "acc": (sum(vals) / len(vals) if vals else None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", default=None, help="Override model name (default: run config)")
    p.add_argument("--layer", type=int, default=None, help="Override layer (default: run config)")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--vector-file", default="vectors/v_mast_mc.pt",
                   help="Vector to steer with, relative to run dir")
    p.add_argument("--label", default="mast", help="Label for the steered variant")
    p.add_argument("--variants", nargs="+", default=["baseline", "steered"],
                   choices=["baseline", "steered"])
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON (default <run-dir>/mc_harness_results.json)")
    args = p.parse_args()

    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    model_name = args.model or cfg["model"]["name"]
    layer = args.layer if args.layer is not None else cfg["model"]["layer"]

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    LOG.info("Model: %s | layer %d | scale %.2f", model_name, layer, args.scale)
    lm = HFLM(pretrained=model_name, dtype="bfloat16", batch_size=args.batch_size,
              parallelize=True)
    hf_model = lm.model

    test_questions = load_test_questions(args.run_dir)
    LOG.info("Test split: %d questions", len(test_questions))

    out = {
        "model": model_name,
        "layer": layer,
        "scale": args.scale,
        "vector_file": str(args.vector_file),
        "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
        "tasks": TASKS,
        "protocol": "harness default: QA primer, summed log-probs, no chat template",
        "variants": {},
    }

    handle = None
    for variant in args.variants:
        if variant == "steered":
            vector = torch.load(args.run_dir / args.vector_file, map_location="cpu")
            layer_module = get_decoder_layer(hf_model, layer)
            handle = layer_module.register_forward_hook(make_steering_hook(vector, args.scale))
            label = args.label
        else:
            label = "baseline"

        LOG.info("=== Variant: %s ===", label)
        results = lm_eval.simple_evaluate(model=lm, tasks=TASKS, log_samples=True,
                                          batch_size=args.batch_size)
        entry = {"full": {}, "test_split": {}, "samples": {}}
        for task in TASKS:
            records = sample_records(results["samples"][task])
            entry["full"][task] = aggregate(records)
            entry["test_split"][task] = aggregate(records, keep=test_questions)
            entry["samples"][task] = records
            LOG.info("%s %s: full=%.4f (n=%d) | test-split=%.4f (n=%d)",
                     label, task,
                     entry["full"][task]["acc"], entry["full"][task]["n"],
                     entry["test_split"][task]["acc"], entry["test_split"][task]["n"])
        out["variants"][label] = entry

        if handle is not None:
            handle.remove()
            handle = None

    dest = args.output or (args.run_dir / "mc_harness_results.json")
    dest.write_text(json.dumps(out, indent=2))
    LOG.info("Saved: %s", dest)


if __name__ == "__main__":
    main()
