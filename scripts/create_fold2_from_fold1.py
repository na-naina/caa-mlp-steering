#!/usr/bin/env python3
"""Create fold2 splits from fold1 (inverse the train/test split)."""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_fold2_from_fold1.py <base_output_dir>")
        print("Example: python scripts/create_fold2_from_fold1.py data/outputs/llama2_7b_chat_2fold_20260126_154949")
        sys.exit(1)

    base = Path(sys.argv[1])
    fold1_splits_file = base / "fold1" / "fold_splits.json"

    if not fold1_splits_file.exists():
        print(f"Error: fold1 splits not found at {fold1_splits_file}")
        sys.exit(1)

    # Read fold1 splits
    with open(fold1_splits_file) as f:
        fold1 = json.load(f)

    # Fold2 train = fold1 test, fold2 test = fold1's original train indices
    fold1_all_train = fold1["steering_pool"] + fold1["train"] + fold1["val"]
    fold2_train_indices = fold1["test"]
    fold2_test_indices = fold1_all_train

    # Split fold2 train into steering_pool/train/val (same sizes as fold1)
    splits = {
        "steering_pool": fold2_train_indices[:50],
        "train": fold2_train_indices[50:250],
        "val": fold2_train_indices[250:],
        "test": fold2_test_indices,
    }

    # Save fold2 splits
    fold2_dir = base / "fold2"
    fold2_dir.mkdir(exist_ok=True)
    with open(fold2_dir / "fold_splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Created fold2 splits at {fold2_dir / 'fold_splits.json'}:")
    print(f"  steering_pool: {len(splits['steering_pool'])}")
    print(f"  train: {len(splits['train'])}")
    print(f"  val: {len(splits['val'])}")
    print(f"  test: {len(splits['test'])}")

    # Verify no overlap
    fold1_test_set = set(fold1["test"])
    fold2_test_set = set(splits["test"])
    overlap = fold1_test_set & fold2_test_set
    if overlap:
        print(f"WARNING: {len(overlap)} indices appear in both test sets!")
    else:
        print("Verified: No overlap between fold1 and fold2 test sets")

    total_tested = len(fold1["test"]) + len(splits["test"])
    print(f"Total unique test examples across both folds: {total_tested}")


if __name__ == "__main__":
    main()
