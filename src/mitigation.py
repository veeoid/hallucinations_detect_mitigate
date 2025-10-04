# mitigation.py
from typing import List, Tuple, Optional
from prompting import system_wrap
from generation import gen_any  # same interface you already use

def _prompt_constrain(q: str, base: str) -> str:
    return system_wrap(f"""
You are repairing a possibly wrong short answer to a question.
RULES:
- Answer in ONE sentence, no preamble, no citations.
- If you are not fully sure, PICK the best single answer and append (confidence: low|medium|high).
- Only return 'Unknown' if you truly cannot pick a single candidate.

Question: {q}
Candidate to repair: {base}
Produce: one-sentence repaired answer.
""")

def _prompt_contrast(q: str, samples: List[str]) -> str:
    joined = "\n".join(f"- {s}" for s in samples[:5])
    return system_wrap(f"""
You will compare candidates and pick ONE best answer.
Candidates:
{joined}

RULES:
- First, think briefly (internally) about conflicts.
- Output ONLY the final answer as ONE short sentence.
- If residual uncertainty remains, append (confidence: low|medium|high).
- Return 'Unknown' ONLY if you cannot reasonably select one.

Question: {q}
Output: final one-sentence answer (optionally with confidence tag).
""")

def _prompt_strict(q: str, base: str) -> str:
    return system_wrap(f"""
You must avoid incorrect factual statements.
- If you are not clearly confident, return 'Unknown'.
- Otherwise answer in ONE sentence.

Question: {q}
Candidate: {base}
Output: one sentence or 'Unknown'.
""")



def mitigate_constrain(question: str, base_answer: str, provider: str, model: str, max_tokens: int = 64) -> str:
    prompt = _prompt_constrain(question, base_answer)
    return gen_any(prompt, provider=provider, model=model, k=1, max_tokens=max_tokens)[0].strip()

def mitigate_contrast(question: str, candidates: List[str], provider: str, model: str, max_tokens: int = 64) -> str:
    prompt = _prompt_contrast(question, candidates)
    return gen_any(prompt, provider=provider, model=model, k=1, max_tokens=max_tokens)[0].strip()

def mitigate_strict(question: str, base_answer: str, provider: str, model: str, max_tokens: int = 48) -> str:
    prompt = _prompt_strict(question, base_answer)
    return gen_any(prompt, provider=provider, model=model, k=1, max_tokens=max_tokens)[0].strip()

def mitigate(question: str,
             base_answer: str,
             samples: List[str],
             severity: str,
             provider: str,
             model: str) -> str:
    """severity ∈ {'medium','high'}"""
    if severity == "medium":
        out = mitigate_constrain(question, base_answer, provider, model)
        return out if out else base_answer
    # severity == 'high'
    out = mitigate_contrast(question, samples, provider, model)
    if not out or out.lower().strip() == "unknown":
        # final fallback: strict abstain allowed
        out2 = mitigate_strict(question, base_answer, provider, model)
        return out2 if out2 else "Unknown"
    return out
