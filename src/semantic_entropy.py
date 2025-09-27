import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from math import log

# load a light embedding model
EMB = SentenceTransformer("intfloat/e5-small-v2")

def cluster_by_meaning(texts, cos_thresh=0.85):
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
    ps = np.array([len(c)/total for c in clusters], dtype=float)
    H = -np.sum(ps * np.log(ps + 1e-12))
    Hmax = log(total) if total > 0 else 1.0
    return float(min(1.0, H / max(1e-9, Hmax)))  # normalized 0..1

def semantic_entropy(samples, cos_thresh=0.85):
    clusters = cluster_by_meaning(samples, cos_thresh)
    return normalized_entropy(clusters, len(samples)), clusters
