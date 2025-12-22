# -*- coding: utf-8 -*-
"""
Pure string utilities for normalization / entropy / similarity.
No external dependencies.
"""

import math
import re
from collections import Counter
from typing import List


_WHITES = re.compile(r"\s+")
_PUNCT  = re.compile(r"[^\w\s\-']")

def normalize_short(s: str) -> str:
    """Lowercase, strip punctuation (keep hyphen/apostrophe), collapse whitespace."""
    if not s:
        return ""
    s = s.strip()
    # drop common wrappers like quotes, trailing periods
    s = s.strip(" \"'`“”’.,;:!?()[]{}<>")
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WHITES.sub(" ", s).strip()
    return s


def is_unknownish(s: str) -> bool:
    if not s:
        return True
    s = s.strip().lower()
    return bool(re.match(r"^(unknown|i( do)?n'?t know|not sure|no idea|unsure)$", s))


def shannon_entropy(strings: List[str]) -> float:
    """Entropy of the distribution over unique strings, in nats normalized to [0,1] by ln(K)."""
    if not strings:
        return 0.0
    counts = Counter(strings)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    H = -sum(p * math.log(p) for p in probs if p > 0)
    K = max(1, len(counts))
    return float(H / math.log(K))


def jaccard_sim(a: str, b: str) -> float:
    """Jaccard over word sets after normalization."""
    a = set(normalize_short(a).split())
    b = set(normalize_short(b).split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union
