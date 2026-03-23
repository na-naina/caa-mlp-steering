#!/usr/bin/env python3
"""Generate fold2 splits (swap train/test) for 2-fold CV."""
import json, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.truthfulqa import TruthfulQADatasetManager

parser = argparse.ArgumentParser()
parser.add_argument("--pool", type=int, required=True)
parser.add_argument("--train", type=int, default=309)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

dataset = TruthfulQADatasetManager(seed=42)
splits = dataset.create_pipeline_splits(
    steering_pool_size=args.pool, train_size=args.train, test_size=0
)

# Fold 2: swap train and test, keep steering pool the same
fold2 = {
    "steering_pool": splits.steering_pool,
    "train": splits.test,
    "test": splits.train,
    "val": [],
}

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w") as f:
    json.dump(fold2, f, indent=2)

print(f"Fold2 splits: pool={len(fold2['steering_pool'])}, train={len(fold2['train'])}, test={len(fold2['test'])}")
