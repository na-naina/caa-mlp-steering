#!/usr/bin/env python3
"""Zero-shot transfer eval: apply a trained steering vector on open-domain QA.

Tests whether a TruthfulQA-trained steering vector generalizes beyond its
training benchmark (reviewer Z6VF's ask). Scored by exact-match / alias
containment against gold answers — no LLM judge, no judge circularity.

Datasets:
  popqa   akariasai/PopQA (long-tail factoids; gold + aliases in possible_answers)
  nq      google-research-datasets/nq_open (natural questions; multiple golds)

Usage:
    python scripts/eval_transfer_qa.py --run-dir data/outputs/multiseed/seed_42 \
        --dataset popqa --n 1000
    python scripts/eval_transfer_qa.py --run-dir <dir> --dataset nq --n 1000 \
        --variants baseline steered
Outputs <run-dir>/transfer_<dataset>.json with per-item records + summary.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import load_causal_model
from src.steering.apply import steering_hook
from src.utils.qa_metrics import contains_match, normalize  # noqa: F401 (normalize kept for import compat)

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


def load_qa(dataset: str, n: int, seed: int):
    from datasets import load_dataset
    if dataset == "popqa":
        ds = load_dataset("akariasai/PopQA")["test"]
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        items = []
        for r in ds:
            golds = json.loads(r["possible_answers"]) if isinstance(r["possible_answers"], str) \
                else list(r["possible_answers"])
            items.append({"question": r["question"], "golds": golds,
                          "meta": {"prop": r.get("prop"), "s_pop": r.get("s_pop")}})
        return items
    elif dataset == "nq":
        ds = load_dataset("google-research-datasets/nq_open")["validation"]
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return [{"question": r["question"], "golds": list(r["answer"]), "meta": {}} for r in ds]
    raise ValueError(dataset)


@torch.no_grad()
def generate(model, tokenizer, device, questions, layer, vector, scale, max_new=48,
             batch_size=8):
    outs = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        prompts = [f"Question: {q}\nAnswer:" for q in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=256).to(device)
        ctx = steering_hook(model, layer, vector, scale=scale) if vector is not None else _null()
        with ctx:
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        for j, g in enumerate(gen):
            text = tokenizer.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(text.split("\n")[0].strip())
        if (i // batch_size) % 10 == 0:
            LOG.info("generated %d/%d", min(i + batch_size, len(questions)), len(questions))
    return outs


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=["popqa", "nq"], required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--model", default=None)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--vector-file", default="vectors/v_mast_mc.pt")
    p.add_argument("--variants", nargs="+", default=["baseline", "steered"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    model_name = args.model or cfg["model"]["name"]
    layer = args.layer if args.layer is not None else cfg["model"]["layer"]

    items = load_qa(args.dataset, args.n, args.seed)
    LOG.info("%s: %d questions | model %s layer %d", args.dataset, len(items), model_name, layer)

    loaded = load_causal_model(model_name, dtype="bfloat16", device_map="auto")
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.primary_device
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    vector = torch.load(args.run_dir / args.vector_file, map_location="cpu")

    out = {"dataset": args.dataset, "n": len(items), "model": model_name, "layer": layer,
           "scale": args.scale, "vector_file": args.vector_file, "variants": {}}
    questions = [it["question"] for it in items]
    for variant in args.variants:
        v = vector if variant == "steered" else None
        preds = generate(model, tokenizer, device, questions, layer, v, args.scale,
                         batch_size=args.batch_size)
        correct = [contains_match(pr, it["golds"]) for pr, it in zip(preds, items)]
        # crude abstention detector: hedges with no gold present
        abstain = [bool(re.search(r"\b(i (don'?t|do not) know|cannot|no reliable|unsure)\b",
                                  pr.lower())) and not c for pr, c in zip(preds, correct)]
        out["variants"][variant] = {
            "em_contains": 100 * sum(correct) / len(correct),
            "abstain_rate": 100 * sum(abstain) / len(abstain),
            "records": [{"question": it["question"], "pred": pr, "correct": c, "abstain": a}
                        for it, pr, c, a in zip(items, preds, correct, abstain)],
        }
        LOG.info("%s: EM(contains)=%.1f%% abstain=%.1f%%", variant,
                 out["variants"][variant]["em_contains"],
                 out["variants"][variant]["abstain_rate"])

    dest = args.run_dir / f"transfer_{args.dataset}.json"
    dest.write_text(json.dumps(out, indent=2))
    LOG.info("Saved: %s", dest)


if __name__ == "__main__":
    main()
