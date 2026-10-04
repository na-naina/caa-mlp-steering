#!/usr/bin/env python3
"""2-fold cross-validation splits over all 817 TruthfulQA questions.

Mirrors the protocol of ITI / TruthX / RaLFiT (2-fold CV, every question is
answered exactly once by a model that never trained on it), while keeping our
pool/train structure inside each training half:

  fold1: the standard seed-s split   pool 100 | train 309 | test 408
  fold2: train-side = fold1.test     pool 100 | train 308 | test 409 (= fold1 pool+train)

Because fold1 is exactly the split `run.py --seed s` builds, existing seed-s
results are the fold1 half of the s-th CV repetition.

Usage:
    python scripts/make_cv2_splits.py --seeds 42 123 456 --out-dir data/splits
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.truthfulqa import TruthfulQADatasetManager  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--pool", type=int, default=100)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        ds = TruthfulQADatasetManager(seed=seed)
        f1 = ds.create_pipeline_splits(steering_pool_size=args.pool, train_size=309, test_size=408)
        fold1 = {"steering_pool": f1.steering_pool, "train": f1.train, "val": [], "test": f1.test}

        side2 = np.array(f1.test)
        np.random.default_rng(seed + 1000).shuffle(side2)
        side2 = side2.tolist()
        fold2 = {"steering_pool": side2[:args.pool], "train": side2[args.pool:], "val": [],
                 "test": sorted(f1.steering_pool + f1.train)}

        for name, f in (("fold1", fold1), ("fold2", fold2)):
            tr = set(f["steering_pool"]) | set(f["train"])
            assert not tr & set(f["test"]), "train/test overlap"
        assert sorted(fold1["test"] + fold2["test"]) == list(range(ds.total_examples))

        for name, f in (("fold1", fold1), ("fold2", fold2)):
            path = args.out_dir / f"cv2_s{seed}_{name}.json"
            path.write_text(json.dumps(f))
            print(f"{path}: pool={len(f['steering_pool'])} train={len(f['train'])} test={len(f['test'])}")


if __name__ == "__main__":
    main()
