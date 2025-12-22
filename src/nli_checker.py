# src/nli_checker.py
from __future__ import annotations
from typing import Dict, List

_nli_model = None


def _get_nli_model():
    """
    Initialize and return the NLI cross encoder model.

    The model is cached in a module level variable so it is loaded only once.
    """
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            # Deberta based NLI cross encoder
            _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformers is not installed. "
                "Please run `pip install sentence-transformers`."
            ) from exc
    return _nli_model


def _normalize_scores(raw_scores) -> List[float]:
    """
    Normalize the raw scores output from the cross encoder to a simple list.

    CrossEncoder.predict usually returns a numpy array of shape (1, 3).
    This helper makes sure we always work with a flat list of floats.
    """
    try:
        # numpy like object
        if hasattr(raw_scores, "shape"):
            if raw_scores.shape == (3,):
                scores = raw_scores
            else:
                scores = raw_scores[0]
            return [float(x) for x in list(scores)]
        # list like
        if isinstance(raw_scores, list) or isinstance(raw_scores, tuple):
            if len(raw_scores) == 3:
                return [float(x) for x in raw_scores]
            if len(raw_scores) == 1:
                inner = raw_scores[0]
                return [float(x) for x in inner]
    except Exception:
        pass

    # Fallback, best effort conversion
    try:
        return [float(x) for x in raw_scores]
    except Exception:
        raise ValueError(f"Could not interpret NLI scores: {raw_scores}")


def check_nli(premise: str, hypothesis: str) -> str:
    """
    Run NLI between a premise and hypothesis and return one of:
    'contradiction', 'entailment', or 'neutral'.

    The base model orders scores as:
        [contradiction, entailment, neutral]

    On top of the argmax label, we apply a simple confidence rule:
    low confidence contradictions are treated as neutral to avoid
    over flagging.
    """
    model = _get_nli_model()
    raw_scores = model.predict([(premise, hypothesis)])
    scores = _normalize_scores(raw_scores)

    if len(scores) != 3:
        raise ValueError(
            f"Expected three NLI scores, got {len(scores)}: {scores}"
        )

    label_mapping = ["contradiction", "entailment", "neutral"]
    prediction_index = int(max(range(len(scores)), key=lambda i: scores[i]))
    predicted_label = label_mapping[prediction_index]

    contradiction_score = float(scores[0])

    # Confidence threshold for contradiction
    # If the model is not clearly confident, treat as neutral
    CONTRADICTION_THRESHOLD = 0.50

    if predicted_label == "contradiction" and contradiction_score < CONTRADICTION_THRESHOLD:
        return "neutral"

    return predicted_label
