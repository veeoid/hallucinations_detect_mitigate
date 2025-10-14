# src/negation_probe.py
from generation import gen_any # Correct: use the main dispatcher
import re

_YN = "Answer yes or no only."

def _yn(s: str) -> str:
    s = s.strip().lower()
    return "yes" if s.startswith("y") else ("no" if s.startswith("n") else "")

# This function now requires provider and model to be passed in
def negation_flip_score(answer: str, provider: str, model: str) -> float:
    """
    Extract first proper-name span from the answer; ask affirmative & negated Y/N.
    Return 1.0 if the model answers YES/YES or NO/NO (self-contradiction/confusion).
    Return 0.0 if it answers YES/NO or NO/YES (coherent).
    Return 0.5 if parsing fails.
    """
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", answer)
    if not m:
        return 0.0
    name = m.group(1)
    q1 = f"Did {name} finish high school? {_YN}"
    q2 = f"Is it true that {name} did not finish high school? {_YN}"
    # Correctly call gen_any with the provider and model
    a1 = gen_any(q1, provider=provider, model=model, k=1, max_tokens=4)[0]
    a2 = gen_any(q2, provider=provider, model=model, k=1, max_tokens=4)[0]
    y1, y2 = _yn(a1), _yn(a2)
    if not y1 or not y2:
        return 0.5
    return 1.0 if y1 == y2 else 0.0