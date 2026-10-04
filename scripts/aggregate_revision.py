#!/usr/bin/env python3
"""Aggregate the October 2026 revision runs (2-fold CV x seeds) into paper tables.

Reads data/outputs/rcv_<method>_s<seed>/fold<k>/<variant>/scale_x/gpt_judge_results.json
(written by scripts/evaluate_with_gpt_judge.py) and reports, per method:

  * Truth%, Info%, and both True*Info aggregations
      - product of marginal rates  T x I   (RaLFiT / TruthX / LoFiT convention)
      - per-item conjunction  % true AND informative  (Lin et al. 2022 "% true and informative")
    pooled over the two folds (= all 817 questions answered once by a model that
    never trained on them), then mean +- s.d. over seeds;
  * per-(seed, fold) cells, for paired win counts between methods;
  * paired question-level bootstrap CIs for method differences;
  * per-category breakdown (ITI-style) pooled over seeds;
  * MC1/MC2 from lm-eval-harness (mc_harness*.json), pooled over folds.

Usage:
    python scripts/aggregate_revision.py --out paper/figures/revision_oct2026
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_ROOT = Path("data/outputs")
JUDGE_FILE = "gpt_judge_results.json"  # --judge open -> open_judge_results.json
RCV = re.compile(r"^rcv_(?P<method>.+)_s(?P<seed>\d+)$")

# (method dir, variant, scale) -> display label
MAIN_VARIANTS = {
    ("baseline", "scale_0.00"): "baseline",
    ("steered", "scale_1.00"): "caa_a1",
    ("steered", "scale_2.00"): "caa_a2",
    ("mlp_mc", "scale_1.00"): "mast",
}
LABELS = {
    "baseline": "LLaMA-2-7B-Chat (no intervention)",
    "caa_a1": "Raw CAA (alpha=1)",
    "caa_a2": "Raw CAA (alpha=2, tuned)",
    "mast": "MAST (lr 5e-4)",
    "mast_lr1e-3": "MAST (lr 1e-3)",
    "mast_lr2e-3": "MAST (lr 2e-3)",
    "dvzero_lr5e-4": "Direct vector, zero init (lr 5e-4)",
    "dvzero_lr1e-3": "Direct vector, zero init (lr 1e-3)",
    "dvzero_lr2e-3": "Direct vector, zero init (lr 2e-3)",
    "dvzero_lr5e-3": "Direct vector, zero init (lr 5e-3)",
    "dvcaa_lr5e-4": "Direct vector, CAA init (lr 5e-4)",
    "dvcaa_lr2e-3": "Direct vector, CAA init (lr 2e-3)",
    "loradpo": "LoRA-DPO (W_O+W_down, r=8)",
}


def load_items(path: Path):
    data = json.loads(path.read_text())
    return [(r["question"].strip(), r.get("truth_judgment") == "yes", r.get("info_judgment") == "yes")
            for r in data["results"] if "question" in r]


def discover():
    """-> {method: {seed: {fold: [(question, truth, info), ...]}}}"""
    runs: dict = defaultdict(lambda: defaultdict(dict))
    for d in sorted(OUT_ROOT.glob("rcv_*_s*")):
        m = RCV.match(d.name)
        if not m:
            continue
        method, seed = m["method"], int(m["seed"])
        for fold_dir in sorted(d.glob("fold*")):
            fold = int(fold_dir.name[4:])
            if method == "main":
                for (var, sc), label in MAIN_VARIANTS.items():
                    f = fold_dir / var / sc / JUDGE_FILE
                    if f.exists():
                        runs[label][seed][fold] = load_items(f)
            else:
                f = fold_dir / "mlp_mc" / "scale_1.00" / JUDGE_FILE
                if f.exists():
                    runs[method][seed][fold] = load_items(f)
    return runs


def rates(items):
    t = np.array([x[1] for x in items], float)
    i = np.array([x[2] for x in items], float)
    return {"n": len(items), "truth": 100 * t.mean(), "info": 100 * i.mean(),
            "ti_product": 100 * t.mean() * i.mean(), "ti_conj": 100 * (t * i).mean()}


def summarise(runs):
    table = {}
    for method, seeds in runs.items():
        per_seed, cells = {}, {}
        for seed, folds in seeds.items():
            for fold, items in folds.items():
                cells[f"s{seed}f{fold}"] = rates(items)
            if set(folds) == {1, 2}:
                per_seed[seed] = rates(folds[1] + folds[2])
        row = {"cells": cells, "per_seed": per_seed, "n_seeds_full": len(per_seed)}
        if per_seed:
            for k in ("truth", "info", "ti_product", "ti_conj"):
                v = np.array([s[k] for s in per_seed.values()])
                row[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        table[method] = row
    return table


def paired_bootstrap(runs, a, b, n_boot=10000, seed=0):
    """Question-level paired bootstrap of (a - b) per-item conjunction and product,
    averaged over the seeds both methods share (folds pooled)."""
    shared = sorted(set(runs[a]) & set(runs[b]))
    shared = [s for s in shared if set(runs[a][s]) == {1, 2} and set(runs[b][s]) == {1, 2}]
    if not shared:
        return None
    qs = sorted({q for q, *_ in runs[a][shared[0]][1] + runs[a][shared[0]][2]})
    idx = {q: k for k, q in enumerate(qs)}

    def mat(method):
        T = np.zeros((len(shared), len(qs)))
        I = np.zeros_like(T)
        for si, s in enumerate(shared):
            for q, t, i in runs[method][s][1] + runs[method][s][2]:
                T[si, idx[q]], I[si, idx[q]] = t, i
        return T, I

    Ta, Ia = mat(a)
    Tb, Ib = mat(b)
    rng = np.random.default_rng(seed)
    n = len(qs)
    conj, prod = [], []
    for _ in range(n_boot):
        r = rng.integers(0, n, n)
        conj.append(100 * ((Ta[:, r] * Ia[:, r]).mean() - (Tb[:, r] * Ib[:, r]).mean()))
        prod.append(100 * ((Ta[:, r].mean(1) * Ia[:, r].mean(1)).mean()
                           - (Tb[:, r].mean(1) * Ib[:, r].mean(1)).mean()))
    point_c = 100 * ((Ta * Ia).mean() - (Tb * Ib).mean())
    point_p = 100 * ((Ta.mean(1) * Ia.mean(1)).mean() - (Tb.mean(1) * Ib.mean(1)).mean())
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return {"seeds": shared, "n_questions": n,
            "diff_conj": float(point_c), "ci95_conj": q(conj),
            "diff_product": float(point_p), "ci95_product": q(prod)}


def wins(table, a, b, key="ti_conj"):
    ca, cb = table[a]["cells"], table[b]["cells"]
    shared = sorted(set(ca) & set(cb))
    w = sum(ca[c][key] > cb[c][key] for c in shared)
    return {"cells": len(shared), "a_wins": w, "b_wins": len(shared) - w,
            "per_cell": {c: round(ca[c][key] - cb[c][key], 2) for c in shared}}


def categories(runs, methods):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation")["validation"]
    cat = {q.strip(): c for q, c in zip(ds["question"], ds["category"])}
    out = {}
    for m in methods:
        if m not in runs:
            continue
        b = defaultdict(list)
        for folds in runs[m].values():
            for items in folds.values():
                for q, t, i in items:
                    b[cat.get(q, "?")].append((t, i))
        out[m] = {c: {"n": len(v), "truth": 100 * np.mean([t for t, _ in v]),
                      "info": 100 * np.mean([i for _, i in v]),
                      "ti_conj": 100 * np.mean([t and i for t, i in v]),
                      "ti_product": 100 * np.mean([t for t, _ in v]) * np.mean([i for _, i in v])}
                  for c, v in b.items()}
    return out


def mc_results():
    """Pool lm-eval-harness test-split accuracies over the two folds."""
    acc = defaultdict(lambda: defaultdict(list))  # (label, task) -> seed -> [(acc, n)]
    for f in sorted(OUT_ROOT.glob("rcv_*_s*/fold*/mc_harness*.json")):
        m = RCV.match(f.parent.parent.name)
        if not m:
            continue
        seed = int(m["seed"])
        data = json.loads(f.read_text())
        for label, entry in data.get("variants", data).items():
            if not isinstance(entry, dict) or "test_split" not in entry:
                continue
            name = label if m["method"] == "main" else f"{m['method']}:{label}"
            for task, r in entry["test_split"].items():
                if r.get("acc") is not None:
                    acc[(name, task)][seed].append((r["acc"], r["n"]))
    out = {}
    for (name, task), seeds in acc.items():
        vals = [100 * sum(a * n for a, n in v) / sum(n for _, n in v) for v in seeds.values() if len(v) == 2]
        if vals:
            out.setdefault(name, {})[task] = {"mean": float(np.mean(vals)),
                                              "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                              "n_seeds": len(vals)}
    return out


def fmt(row, k):
    return f"{row[k]['mean']:.1f} ± {row[k]['sd']:.1f}" if k in row else "–"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("paper/figures/revision_oct2026"))
    p.add_argument("--judge", choices=["gpt", "open"], default="gpt",
                   help="gpt = fine-tuned GPT-4o-mini judges; open = AllenAI LLaMA-2 judges")
    args = p.parse_args()
    global JUDGE_FILE
    JUDGE_FILE = "gpt_judge_results.json" if args.judge == "gpt" else "open_judge_results.json"
    args.out.mkdir(parents=True, exist_ok=True)

    runs = discover()
    table = summarise(runs)
    order = [m for m in LABELS if m in table] + sorted(m for m in table if m not in LABELS)

    lines = ["| Method | seeds | Truth | Info | T×I (product) | T∧I (per-item) |", "|---|---|---|---|---|---|"]
    for m in order:
        r = table[m]
        lines.append(f"| {LABELS.get(m, m)} | {r['n_seeds_full']} | {fmt(r, 'truth')} | {fmt(r, 'info')} "
                     f"| {fmt(r, 'ti_product')} | {fmt(r, 'ti_conj')} |")
    md = [f"# Revision results ({args.judge} judges; 2-fold CV over 817 questions; mean ± s.d. over seeds)", "", *lines, ""]

    comparisons = {}
    for a, b in [("mast", "dvzero_lr2e-3"), ("mast", "dvcaa_lr5e-4"), ("mast", "dvzero_lr5e-4"),
                 ("mast", "caa_a2"), ("mast", "baseline"), ("mast", "loradpo"),
                 ("mast_lr2e-3", "dvzero_lr2e-3"), ("mast_lr1e-3", "dvzero_lr1e-3"),
                 ("dvzero_lr2e-3", "caa_a2")]:
        if a in table and b in table:
            comparisons[f"{a} - {b}"] = {"wins": wins(table, a, b), "bootstrap": paired_bootstrap(runs, a, b)}
    md += ["## Paired comparisons (cells = seed×fold; bootstrap over questions)", "",
           "| a − b | wins (a/b of cells) | Δ T∧I [95% CI] | Δ T×I [95% CI] |", "|---|---|---|---|"]
    for k, v in comparisons.items():
        bs = v["bootstrap"]
        ci = (f"{bs['diff_conj']:+.1f} [{bs['ci95_conj'][0]:+.1f}, {bs['ci95_conj'][1]:+.1f}] | "
              f"{bs['diff_product']:+.1f} [{bs['ci95_product'][0]:+.1f}, {bs['ci95_product'][1]:+.1f}]") if bs else "– | –"
        md.append(f"| {k} | {v['wins']['a_wins']}/{v['wins']['b_wins']} of {v['wins']['cells']} | {ci} |")

    mc = mc_results()
    if mc:
        md += ["", "## MC1/MC2 (lm-eval-harness, test-split, folds pooled)", "", "| Variant | MC1 | MC2 | seeds |", "|---|---|---|---|"]
        for name, tasks in sorted(mc.items()):
            g = lambda t: f"{tasks[t]['mean']:.1f} ± {tasks[t]['sd']:.1f}" if t in tasks else "–"  # noqa: E731
            md.append(f"| {name} | {g('truthfulqa_mc1')} | {g('truthfulqa_mc2')} | "
                      f"{max(v['n_seeds'] for v in tasks.values())} |")

    cats = categories(runs, ["baseline", "caa_a2", "dvzero_lr2e-3", "mast"])
    if "mast" in cats:
        md += ["", "## Per-category T∧I (pooled over seeds; sorted by MAST)", "",
               "| Category | n/seed | Baseline | Raw CAA α=2 | Direct vec. | MAST |", "|---|---|---|---|---|---|"]
        nseed = max(table["mast"]["n_seeds_full"], 1)
        for c, r in sorted(cats["mast"].items(), key=lambda kv: -kv[1]["ti_conj"]):
            g = lambda m: f"{cats[m][c]['ti_conj']:.1f}" if m in cats and c in cats[m] else "–"  # noqa: E731
            md.append(f"| {c} | {r['n'] // nseed} | {g('baseline')} | {g('caa_a2')} | {g('dvzero_lr2e-3')} | {g('mast')} |")

    (args.out / f"revision_results_{args.judge}.md").write_text("\n".join(md) + "\n")
    (args.out / f"revision_results_{args.judge}.json").write_text(json.dumps(
        {"table": table, "comparisons": comparisons, "mc": mc, "categories": cats}, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
