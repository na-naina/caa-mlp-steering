#!/usr/bin/env python3
"""CAA extraction variants: how much of the mean-difference vector is the
massive-activation artefact, and does a cleaner extraction recover the
supervised direction?

From the CAA pool of a split, computes mean(h+ - h-) at layer L with pooling over
  all       all non-pad tokens (the pipeline's default, includes BOS)
  nobos     all tokens except the first (BOS)
  answer    answer tokens only
  last      last token only
each optionally with the massive-activation dims (LLaMA-2-7B: 1415, 2533) zeroed,
and reports the cosine with a reference supervised vector. Saves each variant
rescaled to the reference norm, for steering.

Usage:
    python scripts/caa_variants.py --splits-file data/splits/cv2_s42_fold1.json \
        --ref data/outputs/rcv_dvzero_lr2e-3_s42/fold1/vectors/optimized_vector.pt --out-dir data/outputs/rx_caavar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.truthfulqa import TruthfulQADatasetManager  # noqa: E402
from src.models.loader import load_causal_model  # noqa: E402
from src.steering.apply import _get_decoder_layer  # noqa: E402

MASSIVE = [1415, 2533]


@torch.no_grad()
def pooled(model, tok, prompts, answers, layer, mode, batch=8):
    store = {}
    hd = _get_decoder_layer(model, layer).register_forward_hook(
        lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    tok.padding_side = "right"
    outs = []
    for s in range(0, len(prompts), batch):
        P, A = prompts[s:s + batch], answers[s:s + batch]
        enc = tok([p + a for p, a in zip(P, A)], return_tensors="pt", padding=True).to(model.device)
        model(**enc)
        h = store["h"].float()
        am = enc["attention_mask"].clone()
        for i, p in enumerate(P):
            n_p = len(tok(p)["input_ids"])
            n = int(am[i].sum())
            mask = torch.zeros_like(am[i])
            if mode == "all":
                mask[:n] = 1
            elif mode == "nobos":
                mask[1:n] = 1
            elif mode == "answer":
                mask[n_p:n] = 1
            elif mode == "last":
                mask[n - 1] = 1
            outs.append((h[i] * mask.unsqueeze(-1)).sum(0) / mask.sum())
    hd.remove()
    return torch.stack(outs).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--splits-file", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True, help="supervised reference vector (.pt)")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    sd = json.loads(args.splits_file.read_text())
    ds = TruthfulQADatasetManager(seed=42)
    prompts, pos, neg = [], [], []
    for idx in sd["steering_pool"]:
        item = ds.get_item(int(idx))
        bad = item.get("incorrect_answers") or []
        if not bad:
            continue
        prompts.append(f"Question: {item['question'].strip()}\nAnswer:")
        pos.append(" " + (item.get("best_answer") or item["correct_answers"][0]))
        neg.append(" " + bad[0])

    loaded = load_causal_model(args.model, dtype="bfloat16", device_map="auto")
    model, tok = loaded.model, loaded.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    ref = torch.load(args.ref, map_location="cpu").float().flatten()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for mode in ("all", "nobos", "answer", "last"):
        v = (pooled(model, tok, prompts, pos, args.layer, mode) - pooled(model, tok, prompts, neg, args.layer, mode)).mean(0)
        for clean in (False, True):
            w = v.clone()
            if clean:
                w[MASSIVE] = 0
            name = mode + ("_clean" if clean else "")
            res[name] = {"norm": float(w.norm()), "massive_share": float((v[MASSIVE] ** 2).sum() / (v ** 2).sum()),
                         "cos_ref": float(torch.nn.functional.cosine_similarity(w, ref, dim=0))}
            torch.save(w / w.norm() * ref.norm(), args.out_dir / f"{name}.pt")
            print(f"{name:14s} norm {res[name]['norm']:7.3f}  massive-dim share {res[name]['massive_share']:.3f}  "
                  f"cos(supervised) {res[name]['cos_ref']:+.3f}", flush=True)
    (args.out_dir / "caa_variants.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
