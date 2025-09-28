# src/semantic_entropy.py

import numpy as np
# NOTE: sentence-transformers can be installed in different layouts; import lazily below
# from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity(a):
    """
    Compute cosine similarity matrix for a dense 2D numpy array.
    a: shape (n_samples, n_features)
    Returns: shape (n_samples, n_samples)
    """
    a = np.asarray(a)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, a_norm.T)
from math import log


def _get_sentence_transformer():
    """Lazily import SentenceTransformer and return the class.
    Raises ImportError with a helpful message if it's not available in the current env.
    """
    try:
        # standard import
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception:
        # re-raise a clearer error with guidance
        raise ImportError(
            "SentenceTransformer is not available in this Python environment.\n"
            "Install it into the active venv with: python -m pip install sentence-transformers"
        )


def cluster_by_meaning(texts, cos_thresh=0.85):
    """
    Cluster texts by their semantic meaning using cosine similarity.
    We need this to compute semantic entropy which helps measure diversity.
    Explanation: texts in the same cluster have pairwise cosine similarity >= cos_thresh.
    Returns a list of clusters, each cluster is a list of indices into texts.
    """
    # instantiate the embedding model lazily so importing this module doesn't fail
    ST = _get_sentence_transformer()
    EMB = ST("intfloat/e5-small-v2")
    emb = EMB.encode(texts, normalize_embeddings=True)
    S = cosine_similarity(emb)
    clusters = []
    used = set()
    for i in range(len(texts)):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i+1, len(texts)):
            if j not in used and S[i, j] >= cos_thresh:
                group.append(j)
                used.add(j)
        clusters.append(group)
    return clusters

def normalized_entropy(clusters, total):
    """
    Compute the normalized entropy of the cluster size distribution.
    This is needed to measure diversity of generated samples.
    clusters: list of clusters, each cluster is a list of indices
    total: total number of items clustered
    Returns a float in [0, 1], where 0 means all items in one cluster, 1 means all items in their own cluster.
    """
    if total <= 1:
        return 0.0
    ps = np.array([len(c)/total for c in clusters], dtype=float)
    H = -np.sum(ps * np.log(ps + 1e-12))
    # clamp numerical noise
    H = max(0.0, float(H))
    Hmax = log(total)
    return float(min(1.0, H / max(1e-9, Hmax)))  # normalized 0..1

def semantic_entropy(samples, cos_thresh=0.85):
    """
    Compute the semantic entropy of a list of text samples.
    In terms of hallucinations, larger the divsersity, the more likely some samples are hallucinated.
    Returns a tuple (semantic_entropy_score, clusters)
    where semantic_entropy_score is in [0, 1], higher means more diverse.
    """
    clusters = cluster_by_meaning(samples, cos_thresh)
    return normalized_entropy(clusters, len(samples)), clusters
