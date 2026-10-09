#!/usr/bin/env python3
"""Out-of-benchmark factuality check on SimpleQA Verified with EXISTING TruthfulQA steering vectors.

SimpleQA Verified (Haas et al. 2025, arXiv:2509.07968): 1,000 short-form fact-seeking questions with a
single indisputable gold answer, a de-duplicated / re-verified subset of OpenAI's SimpleQA (Wei et al. 2024).
Grading is reference-based (CORRECT / INCORRECT / NOT_ATTEMPTED) by scripts/grade_simpleqa.py, so it does not
depend on the TruthfulQA-fine-tuned judges.

Generation only, no retraining. Prompt = the TruthfulQA six-shot QA preset used for training and the main
evaluation (src/prompts/truthfulqa_presets.py; wrapped in the chat template when TQA_CHAT_TEMPLATE is set),
so the steering vector is applied in the activation context it was trained in; its "I have no comment."
exemplar makes NOT_ATTEMPTED a natural outcome. Greedy decoding, 64 new tokens, cut at the first blank
line / next "Q:".

Variants: baseline | mast (<mast-dir>/vectors/v_mlp_mc.pt) | dv (<dv-dir>/vectors/optimized_vector.pt)
          | loradpo (<lora-dir>/lora_adapter, merged)

    python scripts/eval_simpleqa.py --model meta-llama/Llama-2-7b-chat-hf --layer 8 \
        --mast-dir data/outputs/rcv_main_s42/fold1 --dv-dir data/outputs/rcv_dvzero_lr2e-3_s42/fold1 \
        --lora-dir data/outputs/rcv_loradpo_s42/fold1 --out data/outputs/sqv_llama_s42f1
Writes <out>/generations.json (per-item predictions for every variant).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.loader import load_causal_model  # noqa: E402
from src.prompts.truthfulqa_presets import format_prompt  # noqa: E402
from src.steering.apply import steering_hook  # noqa: E402

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
LOG = logging.getLogger("simpleqa")
DATA = ROOT / "data/simpleqa_verified/simpleqa_verified.csv"
URL = "https://huggingface.co/datasets/stalkermustang/SimpleQA-Verified/resolve/main/simpleqa_verified.csv"


def load_items(n: int | None):
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    rows = list(csv.DictReader(open(DATA, newline="", encoding="utf-8")))
    rows = rows[:n] if n else rows
    return [{"id": r["original_index"], "question": r["problem"].strip(), "target": r["answer"].strip(),
             "topic": r.get("topic"), "answer_type": r.get("answer_type")} for r in rows]


@torch.no_grad()
def gen(model, tok, items, layer, vec, scale, bs):
    outs = []
    dev = next(model.parameters()).device
    for i in range(0, len(items), bs):
        batch = items[i:i + bs]
        enc = tok([format_prompt(it["question"], preset="qa") for it in batch],
                  return_tensors="pt", padding=True).to(dev)
        with steering_hook(model, layer, vec, scale=scale):
            g = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in g:
            t = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(t.split("\n\n")[0].split("\nQ:")[0].strip())
        if (i // bs) % 20 == 0:
            LOG.info("  %d/%d", min(i + bs, len(items)), len(items))
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--mast-dir", type=Path)
    ap.add_argument("--dv-dir", type=Path)
    ap.add_argument("--lora-dir", type=Path)
    ap.add_argument("--alpha", type=float, default=1.0, help="scale for the vector variants")
    ap.add_argument("--n", type=int, default=None, help="first n questions (default: all 1,000)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    items = load_items(a.n)
    loaded = load_causal_model(a.model, dtype="bfloat16", device_map="auto")
    model, tok = loaded.model.eval(), loaded.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / "generations.json"
    res = json.loads(dest.read_text()) if dest.exists() else {
        "model": a.model, "layer": a.layer, "alpha": a.alpha, "n": len(items), "items": items, "preds": {}}
    variants = [("baseline", None)]
    if a.mast_dir:
        variants.append(("mast", a.mast_dir / "vectors/v_mlp_mc.pt"))
    if a.dv_dir:
        variants.append(("dv", a.dv_dir / "vectors/optimized_vector.pt"))
    for name, vf in variants:
        if name in res["preds"]:
            continue
        vec = torch.load(vf, map_location="cpu").float() if vf else None
        LOG.info("variant %s (%s)", name, vf)
        res["preds"][name] = gen(model, tok, items, a.layer, vec, a.alpha, a.bs)
        res.setdefault("sources", {})[name] = str(vf) if vf else None
        dest.write_text(json.dumps(res, indent=1))
    if a.lora_dir and "loradpo" not in res["preds"]:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(a.lora_dir / "lora_adapter")).merge_and_unload().eval()
        LOG.info("variant loradpo (%s)", a.lora_dir)
        res["preds"]["loradpo"] = gen(model, tok, items, a.layer, None, 0.0, a.bs)
        res.setdefault("sources", {})["loradpo"] = str(a.lora_dir / "lora_adapter")
        dest.write_text(json.dumps(res, indent=1))
    LOG.info("saved %s", dest)


if __name__ == "__main__":
    main()
