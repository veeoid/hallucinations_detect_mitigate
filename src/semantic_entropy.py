# -*- coding: utf-8 -*-
"""
Soft clustering without external embeddings. Good enough for short answers.
- Groups strings if they share high Jaccard similarity over word sets.
"""

from typing import List, Set


def _words(s: str) -> Set[str]:
    return set(s.split())


def _sim(a: str, b: str) -> float:
    A, B = _words(a), _words(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union


def cluster_strings_soft(strings: List[str], cos_like_thresh: float = 0.68) -> List[List[str]]:
    """
    Greedy clustering by Jaccard sim over word sets.
    cos_like_thresh ~ “are these basically the same short answer?”
    """
    clusters: List[List[str]] = []
    used = set()
    for i, s in enumerate(strings):
        if i in used:
            continue
        cluster = [s]
        used.add(i)
        for j in range(i + 1, len(strings)):
            if j in used:
                continue
            if _sim(s, strings[j]) >= cos_like_thresh:
                cluster.append(strings[j])
                used.add(j)
        clusters.append(cluster)
    return clusters
