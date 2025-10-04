# src/select_base.py
from semantic_entropy import cluster_by_meaning

UNKNOWNISH = ("unknown", "n/a", "idk", "i don't know", "not sure")

def _is_unknownish(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("unknown") or any(u in s for u in UNKNOWNISH)

def choose_base_answer(samples, cos_thresh=0.85):
    clusters = cluster_by_meaning(samples, cos_thresh)
    if not clusters:
        return -1, "Unknown.", True

    cluster = max(clusters, key=len)
    base_idx = cluster[0]
    base_answer = (samples[base_idx] or "").strip()

    if _is_unknownish(base_answer):
        # if *all* are unknown-ish, accept Unknown (no mitigation needed)
        if all(_is_unknownish(s) for s in samples):
            return base_idx, "Unknown.", False
        # otherwise force mitigation attempt to try repairing
        return base_idx, base_answer, True

    return base_idx, base_answer, False
