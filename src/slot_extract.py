from collections import Counter
import re
from typing import List, Tuple, Dict, Optional
from generation import gen_ollama

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

def _extract_years(text: str) -> List[str]:
    return [y for y in _YEAR_RX.findall(text or "")]

def _extract_ints(text: str) -> List[str]:
    vals = []
    for s in _INT_RX.findall(text or ""):
        vals.append(s.replace(",", ""))
    return vals

def _majority_share(items: List[str]) -> float:
    if not items: return 1.0
    c = Counter(items)
    return max(c.values()) / len(items)

def slot_agreement(samples: List[str]) -> Tuple[float, Dict[str,float]]:
    names_all, years_all, ints_all = [], [], []
    for s in samples:
        names_all.append("|".join(sorted(_extract_names(s))))
        years_all.append("|".join(sorted(_extract_years(s))))
        ints_all.append("|".join(sorted(_extract_ints(s))))

    names_share = _majority_share(names_all)
    years_share = _majority_share(years_all)
    ints_share  = _majority_share(ints_all)

    present = []
    if any(_extract_names(s) for s in samples): present.append(names_share)
    if any(_extract_years(s) for s in samples): present.append(years_share)
    if any(_extract_ints(s)  for s in samples): present.append(ints_share)

    overall = min(present) if present else 1.0
    return overall, {"names": names_share, "years": years_share, "numbers": ints_share}

def _extract_main_name(answer: str) -> Optional[str]:
    for m in _NAME_RX.finditer(answer or ""):
        first = m.group(1).split()[0]
        if first in _IGNORE_1: continue
        return m.group(1).strip()
    return None

def slot_stability_name(
    question: str,
    base_answer: str,
    model: str = "llama3:instruct",
    provider: str = "ollama",       # NEW: which backend to use
    trials: int = 5,
    max_tokens: int = 20
) -> float:
    """
    Ask the model to extract JUST the main PERSON name from (question, base_answer)
    multiple times and measure stability. Uses the chosen provider (ollama or groq).
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

    # Use unified generator (respects provider)
    from generation import gen as gen_any
    outs = gen_any(prompt, provider=provider, model=model, k=trials, temps=(0.2,), max_tokens=max_tokens)
    outs = [o.strip().splitlines()[0].strip().strip(".").lower() for o in outs]
    cnt = Counter(outs)
    most = cnt.most_common(1)[0][1] if cnt else 0
    return most / max(1, len(outs))