#!/usr/bin/env python3
"""Smoke tests for the multi-task contrastive dataset managers (CPU-only).

Run with pytest:
    .venv/bin/python -m pytest scripts/tests/ -q
or as a plain script:
    .venv/bin/python scripts/tests/test_task_datasets.py

Covers:
  1. TruthfulQA splits for seed 42 are byte-identical to the historical
     pipeline (ground truth: data/outputs/multiseed/seed_42/metadata/splits.json).
  2. The task registry resolves the default task to TruthfulQADatasetManager.
  3. PopQA: construction, negative mining invariants, splits, prompt format,
     determinism.
  4. Behavior A/B (CAA repo): download+parse, mc targets, both embedded
     choice formats, held-out test file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.behavior_ab import BehaviorABDataset
from src.data.contrastive_task import create_task_dataset, get_split_config
from src.data.popqa import PopQAContrastiveDataset
from src.data.truthfulqa import TruthfulQADatasetManager
from src.utils.qa_metrics import normalize

TQA_CACHE = str(REPO_ROOT / "cache" / "datasets")
GROUND_TRUTH_SPLITS = (
    REPO_ROOT / "data" / "outputs" / "multiseed" / "seed_42" / "metadata" / "splits.json"
)

# Known seed-42 values for the llama2_7b_chat_L8_bn8 default recipe
# (pool=100 / train=309 / test=408), captured from the multiseed runs.
EXPECTED_SIZES = {"steering_pool": 100, "train": 309, "test": 408}
EXPECTED_TEST_HEAD = [791, 200, 605, 697, 715]
EXPECTED_POOL_HEAD = [127, 787, 66, 658, 354]
EXPECTED_TRAIN_HEAD = [806, 482, 101, 219, 71]


def test_truthfulqa_splits_unchanged():
    """Default-task seed-42 splits must match the historical pipeline exactly."""
    dataset = TruthfulQADatasetManager(cache_dir=TQA_CACHE, seed=42)
    splits = dataset.create_pipeline_splits(
        steering_pool_size=100, train_size=309, test_size=408
    )

    assert len(splits.steering_pool) == EXPECTED_SIZES["steering_pool"]
    assert len(splits.train) == EXPECTED_SIZES["train"]
    assert len(splits.test) == EXPECTED_SIZES["test"]
    assert splits.test[:5] == EXPECTED_TEST_HEAD, splits.test[:5]
    assert splits.steering_pool[:5] == EXPECTED_POOL_HEAD, splits.steering_pool[:5]
    assert splits.train[:5] == EXPECTED_TRAIN_HEAD, splits.train[:5]

    if GROUND_TRUTH_SPLITS.exists():
        with GROUND_TRUTH_SPLITS.open() as f:
            gt = json.load(f)
        assert splits.steering_pool == gt["steering_pool"]
        assert splits.train == gt["train"]
        assert splits.test == gt["test"]


def test_registry_default_is_truthfulqa():
    """Empty/absent task block must resolve to TruthfulQADatasetManager."""
    config = {"truthfulqa": {"cache_dir": TQA_CACHE}}
    dataset = create_task_dataset(config, seed=42)
    assert isinstance(dataset, TruthfulQADatasetManager)

    # Same seeded rng flow as direct construction -> identical splits.
    splits = dataset.create_pipeline_splits(
        steering_pool_size=100, train_size=309, test_size=408
    )
    assert splits.test[:5] == EXPECTED_TEST_HEAD

    # Split-config resolution: task.split wins over truthfulqa.split.
    assert get_split_config({"truthfulqa": {"split": {"train": 309}}}) == {"train": 309}
    assert get_split_config(
        {"task": {"split": {"train": 50}}, "truthfulqa": {"split": {"train": 309}}}
    ) == {"train": 50}


def test_popqa_dataset():
    dataset = PopQAContrastiveDataset(subsample=500, cache_dir=TQA_CACHE, seed=42)
    assert dataset.total_examples > 400

    # Splits: correct sizes, disjoint.
    splits = dataset.create_pipeline_splits(
        steering_pool_size=20, train_size=50, test_size=100
    )
    assert (len(splits.steering_pool), len(splits.train), len(splits.test)) == (20, 50, 100)
    all_idx = splits.steering_pool + splits.train + splits.test
    assert len(set(all_idx)) == len(all_idx)

    # Negative mining invariants on every item.
    golds_by_prop: dict = {}
    for i in range(dataset.total_examples):
        item = dataset.get_item(i)
        golds_by_prop.setdefault(item["prop"], set()).add(item["best_answer"])
    for i in range(dataset.total_examples):
        item = dataset.get_item(i)
        mc = item["mc1_targets"]
        assert mc["labels"] == [1, 0]
        assert mc["choices"][0] == item["best_answer"]
        wrong = mc["choices"][1]
        # a- is not an alias of the correct answer...
        alias_keys = {normalize(a) for a in item["correct_answers"]}
        assert normalize(wrong) not in alias_keys
        # ...and is another entity's gold answer from the SAME relation type.
        assert wrong in golds_by_prop[item["prop"]]
        # get_mc_targets round-trips through the question.
        assert dataset.get_mc_targets(item["question"]) == mc
        assert dataset.is_valid_mc(i)

    # Prompt format matches the pipeline convention.
    pos, neg, valid = dataset.build_caa_prompts(splits.steering_pool)
    assert len(pos) == len(neg) == len(valid) == 20
    item0 = dataset.get_item(valid[0])
    assert pos[0] == f"Question: {item0['question']}\nAnswer: {item0['best_answer']}"
    assert neg[0] == (
        f"Question: {item0['question']}\nAnswer: {item0['incorrect_answers'][0]}"
    )

    # Determinism: same seed -> same subsample, negatives, and splits.
    dataset2 = PopQAContrastiveDataset(subsample=500, cache_dir=TQA_CACHE, seed=42)
    assert dataset2.total_examples == dataset.total_examples
    assert [dataset2.get_item(i) for i in range(20)] == [
        dataset.get_item(i) for i in range(20)
    ]
    splits2 = dataset2.create_pipeline_splits(
        steering_pool_size=20, train_size=50, test_size=100
    )
    assert splits2.test == splits.test and splits2.train == splits.train


def test_behavior_ab_dataset():
    dataset = BehaviorABDataset(
        behavior="hallucination", cache_dir=REPO_ROOT / "cache" / "behavior_ab", seed=42
    )
    assert dataset.total_examples > 900  # ~1000 generate items, few unparseable

    for i in range(0, dataset.total_examples, 100):
        item = dataset.get_item(i)
        mc = item["mc1_targets"]
        assert mc["labels"] == [1, 0]
        assert item["best_answer"] == mc["choices"][0]
        assert item["incorrect_answers"] == [mc["choices"][1]]
        # The stem must not still embed the options.
        assert "(A)" not in item["question"] and "Choices:" not in item["question"]
        assert dataset.get_mc_targets(item["question"]) == mc
        assert dataset.is_valid_mc(i)

    splits = dataset.create_pipeline_splits(
        steering_pool_size=100, train_size=309, test_size=408
    )
    assert (len(splits.steering_pool), len(splits.train), len(splits.test)) == (100, 309, 408)

    pos, neg, valid = dataset.build_caa_prompts(splits.steering_pool[:10])
    assert len(pos) == len(neg) == len(valid) == 10
    assert pos[0].startswith("Question: ") and "\nAnswer: " in pos[0]

    # Held-out A/B file from the CAA repo parses too.
    heldout = dataset.heldout_ab_items()
    assert len(heldout) >= 40
    assert all(it["mc1_targets"]["labels"] == [1, 0] for it in heldout)

    # Both embedded-choices formats parse (with and without "Choices:" header).
    with_header = BehaviorABDataset._parse_ab_item(
        {
            "question": "Is the sky green?\n\nChoices:\n (A) No, it is blue.\n (B) Yes, bright green.",
            "answer_matching_behavior": "(B)",
            "answer_not_matching_behavior": "(A)",
        }
    )
    assert with_header == {
        "question": "Is the sky green?",
        "best_answer": "Yes, bright green.",
        "correct_answers": ["Yes, bright green."],
        "incorrect_answers": ["No, it is blue."],
        "mc1_targets": {"choices": ["Yes, bright green.", "No, it is blue."], "labels": [1, 0]},
        "matching_letter": "B",
    }
    without_header = BehaviorABDataset._parse_ab_item(
        {
            "question": "How much do you agree?\n (A) A great deal\n (B) Not at all",
            "answer_matching_behavior": "(A)",
            "answer_not_matching_behavior": "(B)",
        }
    )
    assert without_header is not None
    assert without_header["question"] == "How much do you agree?"
    assert without_header["best_answer"] == "A great deal"


def main() -> int:
    tests = [
        test_truthfulqa_splits_unchanged,
        test_registry_default_is_truthfulqa,
        test_popqa_dataset,
        test_behavior_ab_dataset,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {exc!r}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
