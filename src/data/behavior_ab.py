"""Rimsky-style CAA behavioral A/B datasets for the steering-vector pipeline.

Data source: https://github.com/nrimsky/CAA. Layout (verified July 2026):

  datasets/generate/<behavior>/generate_dataset.json   (~1000 items)
  datasets/test/<behavior>/test_dataset_ab.json        (~50 held-out items)

Each item is ``{"question": ..., "answer_matching_behavior": "(A)",
"answer_not_matching_behavior": "(B)"}`` where the *question field embeds both
options* in its text. Two embedded formats exist:

  "<stem>\\n\\nChoices:\\n (A) <text>\\n (B) <text>"   (e.g. hallucination)
  "<stem>\\n (A) <text>\\n (B) <text>"                  (e.g. sycophancy)

We parse out the stem and the two option texts. Contrastive pairs:
  a+ = the behavior-matching option text
  a- = the other option text

``get_mc_targets`` returns ``{"choices": [a+, a-], "labels": [1, 0]}`` so the
existing MC training/eval logprob machinery applies unchanged (A/B held-out
contrastive accuracy).

Files are downloaded into ``cache_dir`` on first use (never committed).
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from src.data.contrastive_task import ContrastiveTaskDataset

logger = logging.getLogger(__name__)

_RAW_BASE = "https://raw.githubusercontent.com/nrimsky/CAA/main/datasets"

KNOWN_BEHAVIORS = (
    "coordinate-other-ais",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
    "refusal",
    "survival-instinct",
    "sycophancy",
)

# Stem, optional "Choices:" header, then the (A)/(B) option texts.
_AB_PATTERN = re.compile(
    r"(?s)^(?P<stem>.*?)\n+\s*(?:Choices:\s*\n)?\s*\(A\)\s*(?P<a>.*?)"
    r"\n\s*\(B\)\s*(?P<b>.*)$"
)


class BehaviorABDataset(ContrastiveTaskDataset):
    """CAA behavioral dataset (hallucination, sycophancy, corrigibility, ...)."""

    def __init__(
        self,
        behavior: str = "hallucination",
        cache_dir: str | Path = "cache/behavior_ab",
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        if behavior not in KNOWN_BEHAVIORS:
            logger.warning(
                "Behavior '%s' not in known list %s; attempting download anyway",
                behavior,
                KNOWN_BEHAVIORS,
            )
        self.behavior = behavior
        self.cache_dir = Path(cache_dir)

        generate_path = self._ensure_file(
            f"{_RAW_BASE}/generate/{behavior}/generate_dataset.json",
            self.cache_dir / behavior / "generate_dataset.json",
        )
        with generate_path.open() as f:
            raw_items = json.load(f)
        logger.info("Loaded %d '%s' generate items", len(raw_items), behavior)

        self.items = self._parse_items(raw_items)
        self.total_examples = len(self.items)
        logger.info(
            "Parsed %d A/B contrastive items (%d unparseable dropped)",
            self.total_examples,
            len(raw_items) - self.total_examples,
        )

        self._mc_by_question: Dict[str, dict] = {
            self._question_key(item["question"]): item["mc1_targets"]
            for item in self.items
        }

    @staticmethod
    def _ensure_file(url: str, dest: Path) -> Path:
        """Download *url* to *dest* on first use."""
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s -> %s", url, dest)
        tmp = dest.with_suffix(".tmp")
        with urllib.request.urlopen(url, timeout=60) as resp:
            tmp.write_bytes(resp.read())
        # Validate before moving into place.
        json.loads(tmp.read_text())
        tmp.rename(dest)
        return dest

    @classmethod
    def _parse_items(cls, raw_items: List[dict]) -> List[dict]:
        items: List[dict] = []
        for raw in raw_items:
            parsed = cls._parse_ab_item(raw)
            if parsed is None:
                logger.debug("Failed to parse A/B item: %r", raw.get("question", "")[:80])
                continue
            items.append(parsed)
        return items

    @classmethod
    def _parse_ab_item(cls, raw: dict) -> Optional[dict]:
        """Split the embedded-choices question into stem + option texts."""
        match = _AB_PATTERN.match(raw.get("question", ""))
        if not match:
            return None
        stem = match.group("stem").strip()
        options = {"A": match.group("a").strip(), "B": match.group("b").strip()}

        letter = (raw.get("answer_matching_behavior") or "").strip().strip("()")
        if letter not in options or not stem:
            return None
        other = "B" if letter == "A" else "A"

        matching, not_matching = options[letter], options[other]
        if not matching or not not_matching:
            return None
        return {
            "question": stem,
            "best_answer": matching,
            "correct_answers": [matching],
            "incorrect_answers": [not_matching],
            "mc1_targets": {
                "choices": [matching, not_matching],
                "labels": [1, 0],
            },
            "matching_letter": letter,
        }

    def get_item(self, index: int) -> dict:
        return dict(self.items[int(index)])

    def get_mc_targets(self, question: str) -> Optional[dict]:
        """Return (behavior-matching, non-matching) choices for a question stem."""
        return self._mc_by_question.get(self._question_key(question))

    def heldout_ab_items(self) -> List[dict]:
        """Parsed items from the CAA repo's held-out test_dataset_ab.json.

        These live outside this dataset's index space (splits index only into
        generate_dataset.json); use them for an additional fixed held-out eval.
        """
        test_path = self._ensure_file(
            f"{_RAW_BASE}/test/{self.behavior}/test_dataset_ab.json",
            self.cache_dir / self.behavior / "test_dataset_ab.json",
        )
        with test_path.open() as f:
            raw_items = json.load(f)
        return self._parse_items(raw_items)
