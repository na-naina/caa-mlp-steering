#!/usr/bin/env python3
"""Does the supervised TruthfulQA steering direction match the 'truth direction' of
Marks & Tegmark (2023, The Geometry of Truth)?

For each of their true/false statement datasets we take the residual stream at the
output of layer L (the steering layer) on the statement's last token, and report:

  * AUROC of the projection onto each candidate direction (true should score higher):
      v_mast, v_dvzero, v_dvcaa (supervised, trained on TruthfulQA answers only),
      v_caa (mean-difference on TruthfulQA), random, and the mass-mean (MM) truth
      direction of each GoT dataset (held-out: estimated on the *other* datasets);
  * cosine between each supervised direction and each dataset's MM direction.

Forward passes only; no generation, no judges.

Usage:
    python scripts/geometry_of_truth.py --cell data/outputs/rcv_main_s42/fold1 \
        --dv-zero data/outputs/rcv_dvzero_lr2e-3_s42/fold1 --dv-caa data/outputs/rcv_dvcaa_lr2e-3_s42/fold1 \
        --out data/outputs/got_s42f1.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import load_causal_model  # noqa: E402
from src.steering.apply import _get_decoder_layer  # noqa: E402

GOT = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/{}.csv"
DATASETS = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "larger_than", "smaller_than",
            "common_claim_true_false", "companies_true_false", "counterfact_true_false"]


def load_got(name: str, cache: Path, max_n: int):
    import pandas as pd
    f = cache / f"{name}.csv"
    if not f.exists():
        cache.mkdir(parents=True, exist_ok=True)
        f.write_bytes(urllib.request.urlopen(GOT.format(name)).read())
    df = pd.read_csv(f)
    df = df.sample(n=min(max_n, len(df)), random_state=0) if len(df) > max_n else df
    return df["statement"].tolist(), df["label"].astype(int).to_numpy()


@torch.no_grad()
def last_token_acts(model, tok, texts, layer, batch=32):
    store = {}
    handle = _get_decoder_layer(model, layer).register_forward_hook(
        lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    out = []
    tok.padding_side = "right"
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], return_tensors="pt", padding=True).to(model.device)
        model(**enc)
        last = enc["attention_mask"].sum(1) - 1
        out.append(store["h"][torch.arange(len(last)), last].float().cpu())
    handle.remove()
    return torch.cat(out)


def auroc(scores, labels):
    """Mann-Whitney AUROC (ties get half credit)."""
    scores = np.asarray(scores, dtype=float)
    order = scores.argsort()
    r = np.empty(len(scores))
    r[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):  # average ranks over ties
        m = scores == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--cell", type=Path, required=True, help="rcv_main run dir (v_mlp_mc.pt, v_steered.pt)")
    p.add_argument("--dv-zero", type=Path)
    p.add_argument("--dv-caa", type=Path)
    p.add_argument("--extra", nargs="*", default=[], help="name=path.pt extra directions")
    p.add_argument("--max-n", type=int, default=1500)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    dirs = {"MAST": torch.load(args.cell / "vectors/v_mlp_mc.pt").float(),
            "CAA": torch.load(args.cell / "vectors/v_steered.pt").float()}
    if args.dv_zero:
        dirs["DV-zero"] = torch.load(args.dv_zero / "vectors/optimized_vector.pt").float()
    if args.dv_caa:
        dirs["DV-CAA"] = torch.load(args.dv_caa / "vectors/optimized_vector.pt").float()
    for e in args.extra:
        n, path = e.split("=", 1)
        dirs[n] = torch.load(path).float()
    g = torch.Generator().manual_seed(0)
    dirs["random"] = torch.randn(next(iter(dirs.values())).shape[0], generator=g)
    dirs = {k: v.flatten() / v.norm() for k, v in dirs.items()}

    loaded = load_causal_model(args.model, dtype="bfloat16", device_map="auto")
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    acts, labels = {}, {}
    for name in DATASETS:
        try:
            texts, y = load_got(name, Path("data/got"), args.max_n)
        except Exception as e:  # noqa: BLE001
            print("skip", name, e)
            continue
        acts[name], labels[name] = last_token_acts(model, tok, texts, args.layer), y
        print(name, len(y), flush=True)

    # mass-mean direction per dataset (centred), and a pooled MM direction from all *other* datasets
    mm = {n: acts[n][labels[n] == 1].mean(0) - acts[n][labels[n] == 0].mean(0) for n in acts}
    res = {"layer": args.layer, "auroc": {}, "cos_to_mm": {}, "mm_heldout_auroc": {}}
    for n in acts:
        res["auroc"][n] = {k: auroc((acts[n] @ d).numpy(), labels[n]) for k, d in dirs.items()}
        others = torch.stack([mm[m] / mm[m].norm() for m in mm if m != n]).mean(0)
        res["mm_heldout_auroc"][n] = auroc((acts[n] @ others).numpy(), labels[n])
        res["cos_to_mm"][n] = {k: float(torch.nn.functional.cosine_similarity(d, mm[n], dim=0))
                               for k, d in dirs.items()}
    pooled = torch.stack([mm[m] / mm[m].norm() for m in mm]).mean(0)
    res["cos_to_pooled_mm"] = {k: float(torch.nn.functional.cosine_similarity(d, pooled, dim=0)) for k, d in dirs.items()}
    torch.save(pooled * (torch.load(args.cell / "vectors/v_mlp_mc.pt").float().norm() / pooled.norm()),
               args.out.with_suffix(".mm_dir.pt"))  # GoT truth direction at MAST's norm, for steering
    args.out.write_text(json.dumps(res, indent=1))

    names = list(dirs) + ["MM(held-out)"]
    print(f"\nAUROC at layer {args.layer} (true > false)")
    print(f"{'dataset':26s}" + "".join(f"{k:>12s}" for k in names))
    for n in acts:
        row = [res["auroc"][n][k] for k in dirs] + [res["mm_heldout_auroc"][n]]
        print(f"{n:26s}" + "".join(f"{v:12.3f}" for v in row))
    print("\ncos(direction, pooled GoT mass-mean truth direction):",
          {k: round(v, 3) for k, v in res["cos_to_pooled_mm"].items()})


if __name__ == "__main__":
    main()
