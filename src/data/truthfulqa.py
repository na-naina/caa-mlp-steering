from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TruthfulQASplits:
    """Indices for the legacy CAA/multiple-choice/generation workflow."""

    caa_indices: Set[int]
    mc_indices: List[int]
    generation_indices: List[int]


@dataclass(frozen=True)
class TruthfulQAPipelineSplits:
    """Pipeline-oriented partition used for steering + training/eval."""

    steering_pool: List[int]
    train: List[int]
    val: List[int]
    test: List[int]


class TruthfulQADatasetManager:
    """Utility for sampling TruthfulQA data for CAA extraction and evaluation."""

    def __init__(
        self,
        dataset_name: str = "truthful_qa",
        dataset_config: str = "generation",
        cache_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rng = np.random.default_rng(seed)

        logger.info(
            "Loading TruthfulQA dataset '%s/%s'",
            self.dataset_name,
            self.dataset_config,
        )
        self.dataset: Dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )["validation"]
        self.total_examples = len(self.dataset)
        logger.info("Loaded %d validation examples", self.total_examples)

        self.mc_lookup = self._build_mc_lookup()
        self.valid_mc_indices = [
            idx
            for idx, item in enumerate(self.dataset)
            if self._question_key(item.get("question", "")) in self.mc_lookup
        ]
        logger.info(
            "Found %d items with valid MC targets", len(self.valid_mc_indices)
        )

    def create_pipeline_splits(
        self,
        *,
        steering_pool_size: int,
        train_size: int,
        val_size: int,
        test_size: int,
    ) -> TruthfulQAPipelineSplits:
        """Create non-overlapping splits for steering/training/evaluation."""

        total_requested = (
            steering_pool_size + train_size + val_size + test_size
        )
        if total_requested > self.total_examples:
            raise ValueError(
                "Requested split sizes exceed available dataset size: "
                f"{total_requested} > {self.total_examples}"
            )

        indices = np.arange(self.total_examples)
        self.rng.shuffle(indices)

        cursor = 0

        steering_pool = indices[cursor : cursor + steering_pool_size].tolist()
        cursor += steering_pool_size

        train = indices[cursor : cursor + train_size].tolist()
        cursor += train_size

        val = indices[cursor : cursor + val_size].tolist()
        cursor += val_size

        test = indices[cursor : cursor + test_size].tolist()

        logger.info(
            "Constructed pipeline splits (pool=%d, train=%d, val=%d, test=%d)",
            len(steering_pool),
            len(train),
            len(val),
            len(test),
        )

        return TruthfulQAPipelineSplits(
            steering_pool=steering_pool,
            train=train,
            val=val,
            test=test,
        )

    def build_caa_prompts(
        self,
        indices: Sequence[int],
        *,
        fallback_negative: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[int]]:
        """Construct positive/negative prompts for provided dataset indices."""

        positive, negative = [], []
        valid_indices: List[int] = []

        for raw_idx in indices:
            idx = int(raw_idx)
            if idx < 0 or idx >= self.total_examples:
                continue
            item = self.dataset[idx]
            question = item["question"].strip()
            best_answer = item.get("best_answer") or item["correct_answers"][0]
            incorrect_answers = item.get("incorrect_answers") or []

            negative_answer = self._select_negative_answer(
                incorrect_answers, fallback=fallback_negative
            )
            if negative_answer is None:
                continue

            positive.append(f"Question: {question}\nAnswer: {best_answer}")
            negative.append(f"Question: {question}\nAnswer: {negative_answer}")
            valid_indices.append(idx)

        if not positive:
            raise RuntimeError("No valid CAA prompts could be constructed")

        logger.info("Prepared %d CAA prompt pairs from provided indices", len(positive))
        return positive, negative, valid_indices

    def sample_caa_pairs(
        self,
        num_samples: int,
    ) -> Tuple[List[str], List[str], Set[int]]:
        """Return positive/negative prompts for CAA extraction."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        indices = self.rng.choice(
            self.total_examples,
            size=min(num_samples, self.total_examples),
            replace=False,
        ).tolist()

        positive, negative, valid_indices = self.build_caa_prompts(indices)
        return positive, negative, set(valid_indices)

    def sample_indices(
        self,
        num_samples: int,
        *,
        exclude: Iterable[int] | None = None,
    ) -> List[int]:
        """Sample dataset indices while avoiding the exclusion set."""
        if num_samples <= 0:
            return []

        all_indices = np.arange(self.total_examples)
        mask = np.ones_like(all_indices, dtype=bool)
        if exclude:
            exclude = set(int(i) for i in exclude)
            mask[[i for i in exclude if i < self.total_examples]] = False

        available_indices = all_indices[mask]
        if available_indices.size == 0:
            raise RuntimeError("No examples available after applying exclusions")

        sample_size = min(num_samples, available_indices.size)
        selection = self.rng.choice(
            available_indices, size=sample_size, replace=False
        )
        return selection.tolist()

    def prepare_splits(
        self,
        *,
        caa_indices: Iterable[int],
        mc_samples: int,
        gen_samples: int,
    ) -> TruthfulQASplits:
        """Materialize MC and generation splits avoiding CAA indices."""
        exclude = set(int(i) for i in caa_indices)
        mc_indices = self._sample_mc_indices(mc_samples, exclude)
        exclude.update(mc_indices)
        gen_indices = self.sample_indices(gen_samples, exclude=exclude)
        return TruthfulQASplits(
            caa_indices=set(int(i) for i in caa_indices),
            mc_indices=mc_indices,
            generation_indices=gen_indices,
        )

    def _sample_mc_indices(self, num_samples: int, exclude: Set[int]) -> List[int]:
        if num_samples <= 0:
            return []
        pool = [idx for idx in self.valid_mc_indices if idx not in exclude]
        if not pool:
            logger.warning("No valid MC indices available after exclusions")
            return []
        sample_size = min(num_samples, len(pool))
        selection = self.rng.choice(pool, size=sample_size, replace=False)
        return selection.tolist()

    @staticmethod
    def _has_valid_mc(mc: dict) -> bool:
        if not mc:
            return False
        choices = mc.get("choices", [])
        labels = mc.get("labels", [])
        if not choices or not labels:
            return False
        has_correct = any(label == 1 for label in labels)
        has_incorrect = any(label == 0 for label in labels)
        return has_correct and has_incorrect

    def get_items(self, indices: Sequence[int]) -> List[dict]:
        """Return dataset items for provided indices with MC augmentation."""
        return [self.get_item(int(i)) for i in indices]

    def get_item(self, index: int) -> dict:
        item = dict(self.dataset[int(index)])
        key = self._question_key(item.get("question", ""))
        mc = self.mc_lookup.get(key)
        if mc:
            item["mc1_targets"] = mc
        return item

    def is_valid_mc(self, index: int) -> bool:
        key = self._question_key(self.dataset[int(index)].get("question", ""))
        return key in self.mc_lookup

    def get_mc_targets(self, question: str) -> dict | None:
        """Get MC targets for a given question from the mc_lookup."""
        key = self._question_key(question)
        return self.mc_lookup.get(key)

    def _build_mc_lookup(self) -> dict:
        logger.info("Loading TruthfulQA multiple-choice split for MC targets")
        try:
            mc_dataset = load_dataset(
                self.dataset_name,
                "multiple_choice",
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )["validation"]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load multiple-choice split for TruthfulQA (%s); MC metrics disabled",
                exc,
            )
            return {}
        lookup = {}
        for item in mc_dataset:
            question = item.get("question", "")
            mc_targets = item.get("mc1_targets")
            if not self._has_valid_mc(mc_targets):
                continue
            key = self._question_key(question)
            lookup[key] = mc_targets
        logger.info("Prepared MC lookup with %d entries", len(lookup))
        return lookup

    @staticmethod
    def _select_negative_answer(
        incorrect_answers: Sequence[str], *, fallback: Optional[str] = None
    ) -> Optional[str]:
        if incorrect_answers:
            for answer in incorrect_answers:
                if answer and answer.strip():
                    return answer
        if fallback is not None:
            return fallback
        return None

    @staticmethod
    def _question_key(question: str) -> str:
        return (question or "").strip().casefold()
