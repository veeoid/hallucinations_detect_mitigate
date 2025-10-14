# src/nli_checker.py
from __future__ import annotations
from typing import List, Dict

_nli_model = None

def _get_nli_model():
    """Initializes and returns the NLI model."""
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            # This is a well-regarded model specifically trained for NLI tasks.
            _nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
        except ImportError:
            raise ImportError("SentenceTransformers is not installed. Please run 'pip install sentence-transformers'.")
    return _nli_model

def check_nli(premise: str, hypothesis: str) -> str:
    """
    Checks the relationship between a premise and a hypothesis using an NLI model.

    Returns:
        A string: 'contradiction', 'entailment', or 'neutral'.
    """
    model = _get_nli_model()
    scores = model.predict([(premise, hypothesis)])
    
    # The model outputs scores for [contradiction, entailment, neutral]
    # We find the index of the highest score to determine the label.
    label_mapping = ['contradiction', 'entailment', 'neutral']
    prediction_idx = scores.argmax()
    
    return label_mapping[prediction_idx]