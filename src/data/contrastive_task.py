"""Task-agnostic interface for contrastive steering datasets.

The steering-vector pipeline (run.py + src/steering/training.py) was built
around :class:`~src.data.truthfulqa.TruthfulQADatasetManager`. This module
captures the de-facto interface that pipeline consumers rely on, so the same
recipe (extract CAA vectors -> train steering MLP -> evaluate) can run on any
task that provides contrastive (positive, negative) answer pairs.

A dataset manager must provide:

- ``total_examples`` -- number of items; splits index into ``range(total_examples)``.
- ``rng`` -- a ``numpy.random.Generator`` seeded in ``__init__(seed=...)``.
  ``create_pipeline_splits`` consumes this stream, so construction order matters
  for reproducibility.
- ``create_pipeline_splits(steering_pool_size=, train_size=, test_size=, val_size=)``
  -> object with ``.steering_pool`` / ``.train`` / ``.test`` / ``.val`` index lists.
- ``build_caa_prompts(indices)`` -> ``(positive_prompts, negative_prompts,
  valid_indices)`` where prompts are formatted ``"Question: {q}\\nAnswer: {a}"``.
- ``get_mc_targets(question)`` -> ``{"choices": [...], "labels": [1, 0, ...]}``
  or ``None``; the first correct and first incorrect choice form the training
  pair (see ``src.steering.training._select_mc_answers``).
- ``is_valid_mc(index)`` -> whether the item has usable MC targets.
- ``get_item(index)`` / ``get_items(indices)`` -> item dicts carrying at least
  ``question``, ``best_answer``, ``correct_answers``, ``incorrect_answers`` and
  (when available) ``mc1_targets`` -- the keys consumed by generation training
  and by src/evaluation/truthfulqa.py.

:class:`TruthfulQADatasetManager` satisfies this interface as-is (duck-typed;
it is intentionally NOT refactored so its seeded rng flow stays byte-identical).
New tasks should subclass :class:`ContrastiveTaskDataset`, which supplies
shared split logic and mc1-based default implementations.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSplits:
    """Task-agnostic pipeline partition (mirrors TruthfulQAPipelineSplits)."""

    steering_pool: List[int]
    train: List[int]
    test: List[int]
    val: List[int] = field(default_factory=list)


class ContrastiveTaskDataset(ABC):
    """Base class for contrastive steering task datasets.

    Subclasses must populate ``self.total_examples`` during ``__init__`` (after
    calling ``super().__init__(seed=seed)``) and implement :meth:`get_item` and
    :meth:`get_mc_targets`. Items whose ``mc1_targets`` contain at least one
    correct (label 1) and one incorrect (label 0) choice automatically work
    with the default :meth:`build_caa_prompts` / :meth:`is_valid_mc`.
    """

    total_examples: int

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Splits (identical algorithm to TruthfulQADatasetManager)
    # ------------------------------------------------------------------
    def create_pipeline_splits(
        self,
        *,
        steering_pool_size: int,
        train_size: int,
        test_size: int = 0,
        val_size: int = 0,
    ) -> PipelineSplits:
        """Create non-overlapping splits for steering/training/evaluation.

        If *test_size* is 0, test gets all remaining items after
        steering_pool + train + val are allocated.

        Allocation order: *test is reserved first* from the tail of the
        shuffled index array, then pool / train / val are taken from the
        head. This keeps the test set stable across configurations that vary
        `steering_pool_size` or `train_size` with the same seed.
        """
        fixed = steering_pool_size + train_size + val_size
        if test_size == 0:
            test_size = self.total_examples - fixed

        total_requested = fixed + test_size
        if total_requested > self.total_examples:
            raise ValueError(
                "Requested split sizes exceed available dataset size: "
                f"{total_requested} > {self.total_examples}"
            )

        indices = np.arange(self.total_examples)
        self.rng.shuffle(indices)

        test = indices[self.total_examples - test_size : self.total_examples].tolist()
        remainder = indices[: self.total_examples - test_size]

        cursor = 0
        steering_pool = remainder[cursor : cursor + steering_pool_size].tolist()
        cursor += steering_pool_size

        train = remainder[cursor : cursor + train_size].tolist()
        cursor += train_size

        val = remainder[cursor : cursor + val_size].tolist() if val_size else []

        logger.info(
            "Constructed pipeline splits (pool=%d, train=%d, val=%d, test=%d)",
            len(steering_pool),
            len(train),
            len(val),
            len(test),
        )

        return PipelineSplits(
            steering_pool=steering_pool,
            train=train,
            test=test,
            val=val,
        )

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------
    @abstractmethod
    def get_item(self, index: int) -> dict:
        """Return the item dict for *index* (see module docstring for keys)."""

    def get_items(self, indices: Sequence[int]) -> List[dict]:
        """Return item dicts for provided indices."""
        return [self.get_item(int(i)) for i in indices]

    @abstractmethod
    def get_mc_targets(self, question: str) -> Optional[dict]:
        """Return ``{"choices": [...], "labels": [...]}`` for *question*, or None."""

    def is_valid_mc(self, index: int) -> bool:
        """Whether the item has a usable (correct, incorrect) choice pair."""
        return self._has_valid_mc(self.get_item(int(index)).get("mc1_targets"))

    # ------------------------------------------------------------------
    # Contrastive prompt construction
    # ------------------------------------------------------------------
    def build_caa_prompts(
        self,
        indices: Sequence[int],
        *,
        fallback_negative: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[int]]:
        """Construct positive/negative prompts for provided dataset indices.

        Default implementation pairs the first correct choice (a+) with the
        first incorrect choice (a-) from the item's ``mc1_targets``.
        """
        positive, negative = [], []
        valid_indices: List[int] = []

        for raw_idx in indices:
            idx = int(raw_idx)
            if idx < 0 or idx >= self.total_examples:
                continue
            item = self.get_item(idx)
            mc = item.get("mc1_targets")
            if not self._has_valid_mc(mc):
                if fallback_negative is None:
                    continue
                mc = {
                    "choices": [item.get("best_answer", ""), fallback_negative],
                    "labels": [1, 0],
                }
            choices, labels = mc["choices"], mc["labels"]
            pos_answer = next(c for c, l in zip(choices, labels) if l == 1)
            neg_answer = next(c for c, l in zip(choices, labels) if l == 0)

            question = item["question"].strip()
            positive.append(f"Question: {question}\nAnswer: {pos_answer}")
            negative.append(f"Question: {question}\nAnswer: {neg_answer}")
            valid_indices.append(idx)

        if not positive:
            raise RuntimeError("No valid CAA prompts could be constructed")

        logger.info("Prepared %d CAA prompt pairs from provided indices", len(positive))
        return positive, negative, valid_indices

    @staticmethod
    def _has_valid_mc(mc: Optional[dict]) -> bool:
        if not mc:
            return False
        choices = mc.get("choices", [])
        labels = mc.get("labels", [])
        if not choices or not labels:
            return False
        return any(l == 1 for l in labels) and any(l == 0 for l in labels)

    @staticmethod
    def _question_key(question: str) -> str:
        return (question or "").strip().casefold()


# ----------------------------------------------------------------------
# Task registry
# ----------------------------------------------------------------------
TASK_NAMES = ("truthfulqa", "popqa", "behavior_ab")


def create_task_dataset(config: Dict[str, Any], seed: int):
    """Instantiate the dataset manager described by ``config['task']``.

    ``config`` is the merged run config (base.yaml + model yaml). The optional
    ``task:`` block selects the dataset::

        task:
          name: popqa          # truthfulqa (default) | popqa | behavior_ab
          subsample: 4000      # popqa only
          behavior: hallucination  # behavior_ab only
          cache_dir: cache/datasets
          split: {steering_pool: 100, train: 309, test: 408}

    Without a ``task:`` block this returns a TruthfulQADatasetManager built
    from the legacy ``truthfulqa:`` block -- identical to the historic run.py
    behavior (byte-identical seeded splits).
    """
    task_cfg = config.get("task") or {}
    name = task_cfg.get("name", "truthfulqa")

    if name == "truthfulqa":
        from src.data.truthfulqa import TruthfulQADatasetManager

        tqa_cfg = config.get("truthfulqa", {})
        return TruthfulQADatasetManager(
            dataset_name=tqa_cfg.get("dataset_name", "truthful_qa"),
            dataset_config=tqa_cfg.get("dataset_config", "generation"),
            cache_dir=tqa_cfg.get("cache_dir"),
            seed=seed,
        )
    if name == "popqa":
        from src.data.popqa import PopQAContrastiveDataset

        return PopQAContrastiveDataset(
            subsample=task_cfg.get("subsample", 4000),
            cache_dir=task_cfg.get("cache_dir", "cache/datasets"),
            seed=seed,
        )
    if name == "behavior_ab":
        from src.data.behavior_ab import BehaviorABDataset

        return BehaviorABDataset(
            behavior=task_cfg.get("behavior", "hallucination"),
            cache_dir=task_cfg.get("cache_dir", "cache/behavior_ab"),
            seed=seed,
        )
    raise ValueError(f"Unknown task '{name}' (expected one of {TASK_NAMES})")


def get_split_config(config: Dict[str, Any]) -> Dict[str, int]:
    """Resolve the split-size block: ``task.split`` wins over ``truthfulqa.split``."""
    task_cfg = config.get("task") or {}
    return task_cfg.get("split") or config.get("truthfulqa", {}).get("split", {})
