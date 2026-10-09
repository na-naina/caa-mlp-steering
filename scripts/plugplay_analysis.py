#!/usr/bin/env python3
"""Is MAST plug-and-play? Per model, compare the default recipe with a small sensitivity grid.

Grid (scripts/revision_queue.py --block plugplay; seeds 42/123 x folds 1/2):
  default   lr 5e-4, picked layer, alpha 1       (the model's main run, <main>_s<seed>/fold<k>/mlp_mc)
  lr x0.5   lr 2.5e-4                             (rpp_<key>_mast_lr2.5e-4_s<seed>/fold<k>)
  lr x2     lr 1e-3                               (rpp_<key>_mast_lr1e-3_...; LLaMA: rcv_mast_lr1e-3_...)
  layer     runner-up train-signal layer          (rpp_<key>_mast_L<alt>_...)
  alpha     0.5 / 1.5, default vector             (rpp_<key>_alpha<a>_s<seed>f<k>, generation only)

Reports per model (T x I, product of rates, GPT judges; mean over the cells where every needed config exists):
  default      the recipe as-is
  oracle       best grid config per cell, chosen on that cell's own test scores (upper bound, not legitimate)
  other-fold   config chosen by test score on the OTHER fold of the same seed, evaluated on this fold
               (test-free for the evaluated fold: the cross-fitting used for lr selection in the paper)
  train-signal config chosen by final MC margin accuracy over trained configs (alpha fixed at 1)
plus the per-config means, so the paper can show how flat the neighbourhood of the default is.

    python scripts/plugplay_analysis.py [--keys llama g4b q4b g4e q35 olmo] [--alt llama=10 olmo=11 ...]
Writes paper/figures/revision_oct2026/plugplay_{gpt}.{md,json}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/outputs"
MAIN = {"llama": "rcv_main", "g4b": "rcv_g4bmain", "q4b": "rcv_q4bmain", "g4e": "rcv_g4emain",
        "q35": "rcv_q35main", "olmo": "rcv_olmomain"}
ALT = {"llama": 10, "g4b": 9, "q4b": 9, "g4e": 10, "q35": 11, "olmo": None}  # runner-up layers; override --alt
LABEL = {"llama": "LLaMA-2-7B-Chat", "g4b": "Gemma-3-4B-IT", "q4b": "Qwen3-4B", "g4e": "Gemma-4-E4B-IT",
         "q35": "Qwen3.5-9B", "olmo": "OLMo-3-7B-Instruct"}
CELLS = [(s, f) for s in (42, 123) for f in (1, 2)]


def config_dirs(key: str, s: int, f: int, alt: int | None, judge_file: str):
    """config name -> (judged results file, training_history.json or None)."""
    lr2 = (f"rcv_mast_lr1e-3_s{s}/fold{f}" if key == "llama" else f"rpp_{key}_mast_lr1e-3_s{s}/fold{f}")
    runs = {"default": f"{MAIN[key]}_s{s}/fold{f}",
            "lr x0.5": f"rpp_{key}_mast_lr2.5e-4_s{s}/fold{f}",
            "lr x2": lr2}
    if alt is not None:
        runs[f"layer {alt}"] = f"rpp_{key}_mast_L{alt}_s{s}/fold{f}"
    out = {name: (OUT / r / "mlp_mc/scale_1.00" / judge_file, OUT / r / "training_history.json")
           for name, r in runs.items()}
    for a in ("0.5", "1.5"):
        out[f"alpha {a}"] = (OUT / f"rpp_{key}_alpha{a}_s{s}f{f}/fold1/mlp_mc/scale_1.00" / judge_file, None)
    return out


def ti(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    rows = [r for r in (data.get("results", []) if isinstance(data, dict) else data) if "question" in r]
    t = [r.get("truth_judgment") == "yes" for r in rows]
    i = [r.get("info_judgment") == "yes" for r in rows]
    return 100 * (sum(t) / len(t)) * (sum(i) / len(i)) if rows else None


def train_acc(path: Path | None):
    if path is None or not path.exists():
        return None
    acc = json.loads(path.read_text())["mc"]["accuracy"][-20:]
    return sum(acc) / len(acc)


def analyse(key: str, alt: int | None, judge_file: str):
    scores, sig = {}, {}
    for s, f in CELLS:
        for name, (jf, th) in config_dirs(key, s, f, alt, judge_file).items():
            v = ti(jf)
            if v is not None:
                scores.setdefault(name, {})[(s, f)] = v
                a = train_acc(th)
                if a is not None:
                    sig.setdefault(name, {})[(s, f)] = a
    names = list(scores)
    full = [c for c in CELLS if all(c in scores[n] for n in names)]
    res = {"model": LABEL[key], "configs": {n: {"mean": mean(scores[n].values()), "cells": len(scores[n])}
                                            for n in names},
           "complete_cells": len(full)}
    if "default" not in scores or not full:
        return res
    pick = lambda c, pool, table: max(pool, key=lambda n: table[n][c])  # noqa: E731
    trained = [n for n in names if n in sig and all(c in sig[n] for c in full)]
    other = {(s, f): (s, 3 - f) for s, f in full}
    rows = {"default": [scores["default"][c] for c in full],
            "oracle": [scores[pick(c, names, scores)][c] for c in full],
            "other-fold": [scores[pick(other[c], names, scores)][c] for c in full if other[c] in full],
            "train-signal": [scores[pick(c, trained, sig)][c] for c in full] if trained else []}
    res["selection"] = {k: {"mean": mean(v), "sd": stdev(v) if len(v) > 1 else 0.0, "n": len(v)}
                        for k, v in rows.items() if v}
    res["chosen"] = {"other-fold": [pick(other[c], names, scores) for c in full if other[c] in full],
                     "train-signal": [pick(c, trained, sig) for c in full] if trained else []}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", nargs="+", default=list(MAIN))
    ap.add_argument("--alt", nargs="*", default=[], help="override runner-up layers, e.g. llama=10 olmo=11")
    ap.add_argument("--judge", choices=["gpt", "open"], default="gpt")
    ap.add_argument("--out", type=Path, default=ROOT / "paper/figures/revision_oct2026")
    a = ap.parse_args()
    alt = dict(ALT, **{k: int(v) for k, v in (x.split("=") for x in a.alt)})
    jf = "gpt_judge_results.json" if a.judge == "gpt" else "open_judge_results.json"

    results = {k: analyse(k, alt[k], jf) for k in a.keys}
    md = [f"# Plug-and-play sensitivity of MAST ({a.judge} judges; T x I product; seeds 42/123 x folds 1/2)", "",
          "| Model | cells | default | oracle (grid best) | other-fold choice | train-signal choice |",
          "|---|---|---|---|---|---|"]
    for k, r in results.items():
        sel = r.get("selection", {})
        cell = lambda n: (f"{sel[n]['mean']:.1f} ± {sel[n]['sd']:.1f}" if n in sel else "–")  # noqa: E731
        md.append(f"| {r['model']} | {r['complete_cells']} | {cell('default')} | {cell('oracle')} | "
                  f"{cell('other-fold')} | {cell('train-signal')} |")
    md += ["", "## Per-config means (cells available)", ""]
    for k, r in results.items():
        md.append(f"- **{r['model']}**: " + ", ".join(f"{n} {c['mean']:.1f} ({c['cells']})"
                                                      for n, c in r["configs"].items()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"plugplay_{a.judge}.md").write_text("\n".join(md) + "\n")
    (a.out / f"plugplay_{a.judge}.json").write_text(json.dumps(results, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
