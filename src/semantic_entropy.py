# src/semantic_entropy.py
# -*- coding: utf-8 -*-
"""
Soft clustering based on semantic similarity using sentence embeddings.
"""
from typing import List, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Lazy loader for the sentence transformer model to avoid loading it on every import
_model = None

def _get_model():
    """Initializes and returns the sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Using a small, fast model suitable for this task
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError("SentenceTransformers is not installed. Please run 'pip install sentence-transformers'.")
    return _model

def _embed(texts: List[str]) -> np.ndarray:
    """Encodes a list of texts into normalized embeddings."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def cluster_strings_soft(strings: List[str], similarity_threshold: float = 0.60) -> List[int]:
    """
    Clusters strings based on semantic similarity.

    Args:
        strings: A list of strings to cluster.
        similarity_threshold: The cosine similarity threshold for grouping strings into the same cluster.
    """
    if not strings:
        return []

    corpus_embeddings = _embed(strings)
    
    # --- FINAL TUNING IS HERE ---
    # Lowered the similarity_threshold from 0.75 to 0.60. This is a more robust
    # value for capturing the similarity between very short answers ("No.") and
    # longer, more descriptive ones that carry the same core meaning.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric='cosine',
        linkage='average'
    )
    
    labels = clustering.fit_predict(corpus_embeddings)
    return labels.tolist()