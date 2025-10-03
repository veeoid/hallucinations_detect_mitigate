# src/select_base.py
import numpy as np
from typing import List, Tuple

def _get_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def _embed(texts: List[str]) -> np.ndarray:
    ST = _get_sentence_transformer()
    EMB = ST("intfloat/e5-small-v2")
    return EMB.encode(texts, normalize_embeddings=True)

def _cosine_sim(E: np.ndarray) -> np.ndarray:
    return np.clip(E @ E.T, -1.0, 1.0)

def choose_base_answer(samples: List[str], cos_thresh: float = 0.85) -> Tuple[int, str]:
    """
    Pick base answer among k samples:
      1) cluster by cosine >= cos_thresh
      2) take the largest cluster
      3) return the medoid (most central) from that cluster
    Returns: (index_in_samples, text)
    """
    if len(samples) == 1:
        return 0, samples[0]

    E = _embed(samples)
    S = _cosine_sim(E)

    # simple greedy clustering around each point
    clusters = []
    used = set()
    for i in range(len(samples)):
        if i in used: continue
        grp = [j for j in range(len(samples)) if S[i, j] >= cos_thresh]
        clusters.append(sorted(grp))
        used.update(grp)

    # largest cluster (tie-breaker: higher mean intra-cluster sim)
    sizes = [len(c) for c in clusters]
    best = np.argmax(sizes)
    if sizes.count(sizes[best]) > 1:
        means = [np.mean(S[np.ix_(c, c)]) for c in clusters]
        best = int(np.argmax(means))

    C = clusters[best]

    # medoid: index in C with max average similarity to others in C
    sims = S[np.ix_(C, C)]
    center_local = int(np.argmax(np.mean(sims, axis=1)))
    base_idx = C[center_local]
    return base_idx, samples[base_idx]
