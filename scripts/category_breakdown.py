#!/usr/bin/env python3
"""Per-category TruthfulQA breakdown from stored judge results (ITI-style).

Joins gpt_judge_results.json items (matched by question text) against the
TruthfulQA `category` column and reports per-category Truth%, Info%, and
T&I% (per-item conjunction). Aggregates across multiple run dirs (e.g. the
10 multiseed runs) by pooling judged items.

Usage:
    python scripts/category_breakdown.py data/outputs/mseed_*/mlp_mc/scale_1.00/gpt_judge_results.json
    python scripts/category_breakdown.py <files...> --csv out.csv --min-n 10
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_category_map() -> dict[str, str]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation")["validation"]
    return {q.strip(): c for q, c in zip(ds["question"], ds["category"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--min-n", type=int, default=1,
                   help="Only print categories with at least this many judged items")
    args = p.parse_args()

    cat_map = load_category_map()
    buckets: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    unmatched = 0
    total = 0

    for f in args.files:
        data = json.loads(f.read_text())
        for r in data["results"]:
            total += 1
            cat = cat_map.get(r["question"].strip())
            if cat is None:
                unmatched += 1
                continue
            buckets[cat].append((r["truth_judgment"] == "yes",
                                 r["info_judgment"] == "yes"))

    rows = []
    for cat, items in buckets.items():
        n = len(items)
        truth = sum(t for t, _ in items) / n
        info = sum(i for _, i in items) / n
        conj = sum(t and i for t, i in items) / n
        rows.append((cat, n, 100 * truth, 100 * info, 100 * conj))
    rows.sort(key=lambda r: r[4])

    print(f"{'Category':<42} {'N':>5} {'Truth%':>7} {'Info%':>7} {'T&I%':>7}")
    print("-" * 72)
    for cat, n, t, i, c in rows:
        if n >= args.min_n:
            print(f"{cat:<42} {n:>5} {t:>7.1f} {i:>7.1f} {c:>7.1f}")
    pooled = [x for items in buckets.values() for x in items]
    n = len(pooled)
    print("-" * 72)
    print(f"{'ALL':<42} {n:>5} "
          f"{100*sum(t for t,_ in pooled)/n:>7.1f} "
          f"{100*sum(i for _,i in pooled)/n:>7.1f} "
          f"{100*sum(t and i for t,i in pooled)/n:>7.1f}")
    if unmatched:
        print(f"(unmatched questions: {unmatched}/{total})")

    if args.csv:
        with args.csv.open("w") as fh:
            fh.write("category,n,truth_pct,info_pct,t_and_i_pct\n")
            for cat, n, t, i, c in rows:
                fh.write(f"\"{cat}\",{n},{t:.2f},{i:.2f},{c:.2f}\n")
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
