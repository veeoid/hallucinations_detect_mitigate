# src/negation_probe.py
from __future__ import annotations
from typing import List
from generation import gen_any
from prompting import system_wrap
import re

def extract_claims(question: str, head_answer: str, provider: str, model: str, max_claims: int = 3) -> List[str]:
    """
    Break (question, answer) into atomic factual claims.
    """
    prompt = system_wrap(f"""
Break the following question-and-answer pair into short factual claims.
Instructions: {{Only answer True or False.}}
Question:
{question}

Answer:
{head_answer}

Write 1 to {max_claims} claims.
Each should be a simple factual statement that can be checked as true or false.
Return one claim per line, no bullet points.
""")

    raw = gen_any(prompt, provider=provider, model=model, k=1, max_tokens=200)[0]
    lines = [l.strip() for l in raw.split("\n") if l.strip()]

    claims = []
    for l in lines:
        claim = re.sub(r'^\s*[\d\.\)\-]+\s*', '', l).strip()
        if claim:
            claims.append(claim)
        if len(claims) >= max_claims:
            break

    return claims


def score_claim_truth(claim: str, provider: str, model: str) -> float:
    """
    Score each factual claim:
       True → 0.0
       Unknown → 0.5
       False → 1.0
    """
    prompt = system_wrap(f"""
You are a careful fact checker.

Claim:
{claim}

Classify this claim as exactly one word:
True
False
Unknown
""")

    ans = gen_any(prompt, provider=provider, model=model, k=1, max_tokens=3)[0].lower().strip()

    if ans.startswith("true"):
        return 0.0
    if ans.startswith("false"):
        return 1.0
    return 0.5


def negation_flip_score(question: str, head_answer: str, provider: str, model: str) -> float:
    """
    Claim-level contradiction score in [0,1].
    Lower = more self-consistent.
    Higher = contradictions present.
    """
    try:
        claims = extract_claims(question, head_answer, provider, model)
    except Exception as e:
        print(f"[NegProbe] claim extraction failed: {e!r}")
        claims = []

    # fallback: treat whole answer as a claim
    if not claims:
        claims = [f'"{head_answer}" is a correct answer to "{question}".']

    scores = []
    for c in claims:
        try:
            s = score_claim_truth(c, provider, model)
            scores.append(s)
        except Exception as e:
            print(f"[NegProbe] claim scoring failed: {e!r}")

    if not scores:
        return 0.5

    final_score = sum(scores) / len(scores)
    return min(1.0, max(0.0, final_score))
