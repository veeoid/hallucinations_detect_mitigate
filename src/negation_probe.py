# src/negation_probe.py
from generation import gen_any
import re

_YN = "Answer yes or no only."

def _yn(s: str) -> str:
    s = s.strip().lower()
    return "yes" if s.startswith("y") else ("no" if s.startswith("n") else "")

def negation_flip_score(answer: str, provider: str, model: str) -> float:
    """
    This probe is now more targeted. It only activates if it finds a
    proper name in the answer, as this is where it is most effective.
    """
    # This regex looks for a capitalized word followed by another, which is a
    # decent heuristic for a proper name (e.g., "Bill Gates", "New York").
    m = re.search(r"\b([A-Z][a-z]+(\s+[A-Z][a-z]+)+)\b", answer)
    if not m:
        # If no multi-word proper name is found, this probe is not applicable.
        return 0.0
    
    name = m.group(1)
    
    # Construct two opposing questions about the entity.
    q1 = f"Is the following statement about {name} correct? Yes or No."
    q2 = f"Is the following statement about {name} incorrect? Yes or No."

    a1_raw = gen_any(q1, provider=provider, model=model, k=1, max_tokens=4)[0]
    a2_raw = gen_any(q2, provider=provider, model=model, k=1, max_tokens=4)[0]

    y1, y2 = _yn(a1_raw), _yn(a2_raw)

    if not y1 or not y2:
        return 0.5  # Penalize uncertainty

    # If the model answers "yes" to both or "no" to both, it's a strong sign of confusion.
    if y1 == y2:
        return 1.0
    
    return 0.0