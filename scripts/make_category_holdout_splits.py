#!/usr/bin/env python3
"""Category-holdout splits: train on one half of TruthfulQA's categories,
test on the other half.

Tests whether the learned steering generalizes across *kinds* of falsehoods
(behavior-level transfer) or only within seen categories (content-level).
Produces splits.json files consumable by run.py --splits-file.

Usage:
    python scripts/make_category_holdout_splits.py --out data/splits/cat_holdout
    # then e.g.:
    # python run.py --model llama2_7b_chat_L8_bn8 --stage train-only --seed 42 \
    #     --splits-file data/splits/cat_holdout/fold_A/splits.json --output-dir ...
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool-size", type=int, default=100)
    p.add_argument("--train-size", type=int, default=309)
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation")["validation"]
    by_cat = defaultdict(list)
    for i, c in enumerate(ds["category"]):
        by_cat[c].append(i)

    rng = np.random.default_rng(args.seed)
    cats = sorted(by_cat)
    rng.shuffle(cats)
    # balance halves by question count
    half_a, half_b, na, nb = [], [], 0, 0
    for c in sorted(cats, key=lambda c: -len(by_cat[c])):
        if na <= nb:
            half_a.append(c); na += len(by_cat[c])
        else:
            half_b.append(c); nb += len(by_cat[c])

    for name, train_cats, test_cats in [("fold_A", half_a, half_b),
                                        ("fold_B", half_b, half_a)]:
        train_pool = [i for c in train_cats for i in by_cat[c]]
        test = sorted(i for c in test_cats for i in by_cat[c])
        rng2 = np.random.default_rng(args.seed)
        rng2.shuffle(train_pool)
        pool = train_pool[:args.pool_size]
        train = train_pool[args.pool_size:args.pool_size + args.train_size]
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "splits.json").write_text(json.dumps({
            "steering_pool": pool, "train": train, "test": test, "val": [],
            "train_categories": sorted(train_cats), "test_categories": sorted(test_cats),
        }, indent=1))
        print(f"{name}: pool={len(pool)} train={len(train)} test={len(test)} "
              f"({len(train_cats)} train cats -> {len(test_cats)} held-out cats)")


if __name__ == "__main__":
    main()
