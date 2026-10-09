#!/usr/bin/env python3
"""Judge-free check on TruthfulQA itself: the Jan 2025 binary-choice format (Lin, Hilton & Evans).

Each of the 790 revised questions has one "Best Answer" and one "Best Incorrect Answer"; an item counts as
correct when the model assigns the higher log-likelihood to the Best Answer (sum over answer tokens, as in
lm-eval MC1; the length-normalised variant is reported too). Only each fold's TEST questions are scored, so a
vector/adapter trained on that fold never sees them. Context = the same six-shot QA prompt as generation
(src/prompts/truthfulqa_presets.py, chat-wrapped when TQA_CHAT_TEMPLATE is set); the steering vector is added
at every position, exactly as during generation.

Variants: baseline | mast (<mast-dir>/vectors/v_mlp_mc.pt) | dv (<dv-dir>/vectors/optimized_vector.pt)
          | loradpo (<lora-dir>/lora_adapter, merged)

    python scripts/eval_binary_tqa.py --model meta-llama/Llama-2-7b-chat-hf --layer 8 \
        --splits-file data/splits/cv2_s42_fold1.json --mast-dir data/outputs/rcv_main_s42/fold1 \
        --dv-dir data/outputs/rcv_dvzero_lr2e-3_s42/fold1 --lora-dir data/outputs/rcv_loradpo_s42/fold1 \
        --out data/outputs/bmc_llama_s42f1
Writes <out>/binary_mc.json (per-item log-likelihoods and per-variant accuracy).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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
LOG = logging.getLogger("binary_tqa")
CSV = ROOT / "data/truthfulqa_binary/TruthfulQA.csv"
URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"


def norm(q: str) -> str:
    return " ".join(q.split()).lower()


def load_items(splits_file: Path):
    if not CSV.exists():
        CSV.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, CSV)
    binary = {norm(r["Question"]): r for r in csv.DictReader(open(CSV, newline="", encoding="utf-8"))}
    from datasets import load_dataset
    hf = load_dataset("truthful_qa", "generation")["validation"]  # split indices refer to this order
    test = json.loads(splits_file.read_text())["test"]
    items, missing = [], 0
    for i in test:
        r = binary.get(norm(hf[i]["question"]))
        if r is None:
            missing += 1
            continue
        items.append({"idx": i, "question": r["Question"].strip(), "category": r["Category"],
                      "best": r["Best Answer"].strip(), "best_incorrect": r["Best Incorrect Answer"].strip()})
    LOG.info("%d test questions, %d in the binary set, %d not in it (removed in the 2025 revision)",
             len(test), len(items), missing)
    return items


@torch.no_grad()
def answer_logprob(model, tok, prompt: str, answer: str):
    sep = "" if os.environ.get("TQA_CHAT_TEMPLATE") else " "
    dev = next(model.parameters()).device
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=not os.environ.get("TQA_CHAT_TEMPLATE")).input_ids
    full = tok(prompt + sep + answer, return_tensors="pt",
               add_special_tokens=not os.environ.get("TQA_CHAT_TEMPLATE")).input_ids
    n_p = p_ids.shape[1]
    logits = model(full.to(dev)).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1)
    tgt = full[0, 1:].to(dev)
    tok_lp = lp[torch.arange(len(tgt), device=dev), tgt][n_p - 1:]
    return float(tok_lp.sum()), int(tok_lp.numel())


def score(model, tok, items, layer, vec, scale):
    out = []
    with steering_hook(model, layer, vec, scale=scale):
        for k, it in enumerate(items):
            prompt = format_prompt(it["question"], preset="qa")
            lb, nb = answer_logprob(model, tok, prompt, it["best"])
            li, ni = answer_logprob(model, tok, prompt, it["best_incorrect"])
            out.append({"lp_best": lb, "n_best": nb, "lp_inc": li, "n_inc": ni,
                        "correct": lb > li, "correct_norm": lb / nb > li / ni})
            if k % 100 == 0:
                LOG.info("  %d/%d", k, len(items))
    acc = 100 * sum(r["correct"] for r in out) / len(out)
    acc_n = 100 * sum(r["correct_norm"] for r in out) / len(out)
    return out, acc, acc_n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--splits-file", type=Path, required=True)
    ap.add_argument("--mast-dir", type=Path)
    ap.add_argument("--dv-dir", type=Path)
    ap.add_argument("--lora-dir", type=Path)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    items = load_items(a.splits_file)
    loaded = load_causal_model(a.model, dtype="bfloat16", device_map="auto")
    model, tok = loaded.model.eval(), loaded.tokenizer

    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / "binary_mc.json"
    res = json.loads(dest.read_text()) if dest.exists() else {
        "model": a.model, "layer": a.layer, "alpha": a.alpha, "splits_file": str(a.splits_file),
        "chat_template": os.environ.get("TQA_CHAT_TEMPLATE"), "items": items, "variants": {}}
    variants = [("baseline", None)]
    if a.mast_dir:
        variants.append(("mast", a.mast_dir / "vectors/v_mlp_mc.pt"))
    if a.dv_dir:
        variants.append(("dv", a.dv_dir / "vectors/optimized_vector.pt"))
    for name, vf in variants:
        if name in res["variants"]:
            continue
        vec = torch.load(vf, map_location="cpu").float() if vf else None
        LOG.info("variant %s (%s)", name, vf)
        per, acc, acc_n = score(model, tok, items, a.layer, vec, a.alpha)
        res["variants"][name] = {"source": str(vf) if vf else None, "acc": acc, "acc_norm": acc_n, "per_item": per}
        LOG.info("  %s: acc %.1f (length-normalised %.1f)", name, acc, acc_n)
        dest.write_text(json.dumps(res, indent=1))
    if a.lora_dir and "loradpo" not in res["variants"]:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(a.lora_dir / "lora_adapter")).merge_and_unload().eval()
        LOG.info("variant loradpo (%s)", a.lora_dir)
        per, acc, acc_n = score(model, tok, items, a.layer, None, 0.0)
        res["variants"]["loradpo"] = {"source": str(a.lora_dir / "lora_adapter"), "acc": acc, "acc_norm": acc_n,
                                      "per_item": per}
        LOG.info("  loradpo: acc %.1f (length-normalised %.1f)", acc, acc_n)
        dest.write_text(json.dumps(res, indent=1))
    LOG.info("saved %s", dest)


if __name__ == "__main__":
    main()
