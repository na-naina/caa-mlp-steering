"""PopQA contrastive dataset for the steering-vector pipeline.

Loads akariasai/PopQA (long-tail factoid QA). Each item carries a question,
a gold answer (``obj``), gold aliases (``possible_answers``), a relation type
(``prop``) and a subject (``subj``).

Contrastive pair construction:
  a+ = the gold answer
  a- = a *mined wrong answer*: another entity's gold answer drawn from the
       SAME relation type (same ``prop``), sampled with a seeded rng and
       excluding anything that normalizes to an alias of the correct answer.

Negatives are mined once at construction time with a dedicated rng (derived
from the seed), so ``self.rng`` -- consumed by ``create_pipeline_splits`` --
sees exactly one shuffle regardless of how many negatives were mined.

Evaluation for this task is EM / alias-containment (see
``src.utils.qa_metrics`` and scripts/eval_transfer_qa.py).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset

from src.data.contrastive_task import ContrastiveTaskDataset
from src.utils.qa_metrics import normalize

logger = logging.getLogger(__name__)


class PopQAContrastiveDataset(ContrastiveTaskDataset):
    """PopQA with relation-mined wrong answers as contrastive negatives."""

    def __init__(
        self,
        dataset_name: str = "akariasai/PopQA",
        subsample: int = 4000,
        cache_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir) if cache_dir else None

        logger.info("Loading PopQA dataset '%s'", self.dataset_name)
        raw = load_dataset(
            self.dataset_name,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )["test"]
        logger.info("Loaded %d PopQA items", len(raw))

        # Dedicated rng for subsampling + negative mining: keeps the split rng
        # stream (self.rng) independent of dataset size / mining details.
        mining_rng = np.random.default_rng(seed)

        if subsample and subsample < len(raw):
            keep = np.sort(
                mining_rng.choice(len(raw), size=subsample, replace=False)
            ).tolist()
            raw = raw.select(keep)
            logger.info("Subsampled to %d items", len(raw))

        records = self._parse_records(raw)
        self.items = self._mine_negatives(records, mining_rng)
        self.total_examples = len(self.items)
        logger.info(
            "Prepared %d PopQA contrastive items (%d dropped in mining)",
            self.total_examples,
            len(records) - self.total_examples,
        )

        self._mc_by_question: Dict[str, dict] = {
            self._question_key(item["question"]): item["mc1_targets"]
            for item in self.items
        }

    @classmethod
    def _parse_records(cls, raw) -> List[dict]:
        """Parse rows, dropping duplicate questions (keeps the question ->
        mc-targets lookup unambiguous for ``get_mc_targets``)."""
        records = []
        seen_questions = set()
        duplicates = 0
        for row in raw:
            gold = (row.get("obj") or "").strip()
            question = (row.get("question") or "").strip()
            if not gold or not question:
                continue
            key = cls._question_key(question)
            if key in seen_questions:
                duplicates += 1
                continue
            seen_questions.add(key)
            aliases_raw = row.get("possible_answers") or "[]"
            aliases = (
                json.loads(aliases_raw)
                if isinstance(aliases_raw, str)
                else list(aliases_raw)
            )
            if gold not in aliases:
                aliases = [gold] + aliases
            records.append(
                {
                    "question": question,
                    "gold": gold,
                    "aliases": [a for a in aliases if a and str(a).strip()],
                    "prop": row.get("prop"),
                    "subj": row.get("subj"),
                    "s_pop": row.get("s_pop"),
                }
            )
        if duplicates:
            logger.info("Dropped %d duplicate-question PopQA rows", duplicates)
        return records

    @staticmethod
    def _mine_negatives(records: List[dict], rng: np.random.Generator) -> List[dict]:
        """Attach a mined wrong answer to each record (same-relation sampling)."""
        golds_by_prop: Dict[str, List[str]] = {}
        for rec in records:
            golds_by_prop.setdefault(rec["prop"], []).append(rec["gold"])
        # Deduplicate while preserving order (determinism).
        for prop, golds in golds_by_prop.items():
            golds_by_prop[prop] = list(dict.fromkeys(golds))

        items: List[dict] = []
        for rec in records:
            alias_keys = {normalize(a) for a in rec["aliases"]}
            candidates = [
                g
                for g in golds_by_prop.get(rec["prop"], [])
                if normalize(g) and normalize(g) not in alias_keys
            ]
            if not candidates:
                logger.debug(
                    "No same-relation negative for %r (prop=%s); dropping",
                    rec["question"],
                    rec["prop"],
                )
                continue
            wrong = candidates[int(rng.integers(len(candidates)))]
            items.append(
                {
                    "question": rec["question"],
                    "best_answer": rec["gold"],
                    "correct_answers": list(rec["aliases"]),
                    "incorrect_answers": [wrong],
                    "mc1_targets": {
                        "choices": [rec["gold"], wrong],
                        "labels": [1, 0],
                    },
                    "prop": rec["prop"],
                    "subj": rec["subj"],
                    "s_pop": rec["s_pop"],
                }
            )
        return items

    def get_item(self, index: int) -> dict:
        return dict(self.items[int(index)])

    def get_mc_targets(self, question: str) -> Optional[dict]:
        """Return the (gold, mined-wrong) choice pair for a given question."""
        return self._mc_by_question.get(self._question_key(question))
