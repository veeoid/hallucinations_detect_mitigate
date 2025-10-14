# src/slot_extract.py
from collections import Counter
import re
from typing import List, Tuple, Dict, Optional
from generation import gen_any # Correct: use the main dispatcher

_NAME_RX = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
_IGNORE_1 = {"A","An","The","In","On","At","For","From","To","I","We","You","He","She","It"}

_YEAR_RX = re.compile(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b")
_INT_RX  = re.compile(r"\b\d{1,3}(?:,\d{3})+|\b\d+\b")

def _extract_names(text: str) -> List[str]:
    names = []
    for m in _NAME_RX.finditer(text or ""):
        first = m.group(1).split()[0]
        if first in _IGNORE_1: 
            continue
        names.append(m.group(1).strip().lower())
    return names

# (No changes to _extract_years, _extract_ints, _majority_share, slot_agreement, _extract_main_name)
# ... functions omitted for brevity ...

def slot_stability_name(
    question: str,
    base_answer: str,
    provider: str, # This function already accepted the provider
    model: str,    # and model
    trials: int = 5,
    max_tokens: int = 20
) -> float:
    """
    Ask the model to extract JUST the main PERSON name from (question, base_answer)
    multiple times and measure stability. Uses the chosen provider.
    """
    name = _extract_main_name(base_answer)
    if not name:
        return 1.0

    prompt = (
        "Extract ONLY the main PERSON name mentioned in the draft answer below.\n"
        "If unsure, output NONE.\n\n"
        f"Question:\n{question}\n\nDraft answer:\n{base_answer}\n\n"
        "Main PERSON:"
    )

    # Correctly call gen_any
    outs = gen_any(prompt, provider=provider, model=model, k=trials, temperature=0.2, max_tokens=max_tokens)
    outs = [o.strip().splitlines()[0].strip().strip(".").lower() for o in outs]
    cnt = Counter(outs)
    most = cnt.most_common(1)[0][1] if cnt else 0
    return most / max(1, len(outs))