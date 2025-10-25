# src/mitigation.py
from typing import List
from prompting import system_wrap
from generation import gen_any
import re

def _prompt_proposer(q: str, samples: List[str]) -> str:
    unique_samples = sorted(list(set(s.strip().rstrip('.') for s in samples if s and len(s) > 10)))
    if not unique_samples:
        unique_samples = sorted(list(set(s.strip().rstrip('.') for s in samples if s)))
        
    candidate_list = "\n".join(f"- \"{s}\"" for s in unique_samples)

    return system_wrap(f"""
You are an assistant. Select the single best answer for the user's question from the candidates provided.

Question:
{q}

Candidates:
{candidate_list}

**CRITICAL**: Your response must be ONLY the text of the best candidate answer. Do not add any other words.

Best Answer:
""")

def _prompt_skeptic_final(q: str, proposed_answer: str) -> str:
    """
    This is the final, hardened skeptic prompt. It is maximally restrictive
    to prevent the model from hallucinating reasons to reject a correct answer.
    """
    return system_wrap(f"""
You are a meticulous fact-checker. Your only goal is to determine if the proposed answer is factually correct.

Question:
{q}

Proposed Answer:
"{proposed_answer}"

YOUR TASK:
Is the proposed answer a factually correct and reliable answer to the question?
- If the answer is completely accurate, respond with the single word: "Yes".
- If the answer is wrong or misleading in any way, respond with the single word: "No".

Your entire response must be only "Yes" or "No".
""")

def _clean_final_answer(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'^\s*[\d\.\-]+\s*', '', text).strip()
    return cleaned

def mitigate(question: str,
             base_answer: str,
             samples: List[str],
             severity: str,
             provider: str,
             model: str) -> str:
    """
    The final, hardened mitigation function using a Proposer/Skeptic debate.
    """
    if severity == 'low':
        return base_answer

    proposer_prompt = _prompt_proposer(question, samples)
    proposed_answer = gen_any(
        proposer_prompt,
        provider=provider,
        model=model,
        k=1,
        max_tokens=100
    )[0].strip()

    if not proposed_answer or "unknown" in proposed_answer.lower():
        return "Unknown"

    skeptic_prompt = _prompt_skeptic_final(question, proposed_answer)
    skeptic_analysis = gen_any(
        skeptic_prompt,
        provider=provider,
        model=model,
        k=1,
        max_tokens=3
    )[0].strip()

    if "yes" in skeptic_analysis.lower():
        return _clean_final_answer(proposed_answer)
    else:
        print(f"[TRACE] Skeptic rejected the answer. Analysis: \"{skeptic_analysis}\"")
        return "Unknown"