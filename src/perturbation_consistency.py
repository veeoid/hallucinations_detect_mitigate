# src/perturbation_consistency.py
from typing import List, Tuple
import numpy as np

# reuse the lazy loader pattern from your entropy module
def _get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception:
        raise ImportError("Install with: pip install sentence-transformers")

def _embed(texts: List[str]):
    ST = _get_sentence_transformer()
    EMB = ST("intfloat/e5-small-v2")
    return EMB.encode(texts, normalize_embeddings=True)

def semantic_match(a: str, b: str, cos_thresh: float = 0.85) -> float:
    embs = _embed([a, b])
    sim = float(np.dot(embs[0], embs[1]))
    return 1.0 if sim >= cos_thresh else max(0.0, (sim - (cos_thresh-0.15)) / 0.15)

def consistency_score(original_answer: str, paraphrase_answers: List[str]) -> float:
    """Return mean semantic match [0..1] between original and paraphrase answers."""
    if not paraphrase_answers:
        return 1.0
    scores = [semantic_match(original_answer, p) for p in paraphrase_answers]
    return float(np.mean(scores))
