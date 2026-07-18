"""Shared answer-matching helpers for open-domain QA evaluation.

SQuAD-style normalization plus alias-containment matching, used by the
zero-shot transfer eval (scripts/eval_transfer_qa.py) and the PopQA
contrastive dataset (src/data/popqa.py).
"""
from __future__ import annotations

import re
import string
from typing import Sequence


def normalize(s: str) -> str:
    """SQuAD-style answer normalization (lowercase, strip punctuation/articles)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_match(pred: str, golds: Sequence[str]) -> bool:
    """True if any normalized gold answer is contained in the normalized prediction."""
    p = normalize(pred)
    return any(normalize(g) in p for g in golds if g and normalize(g))
