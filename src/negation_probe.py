# src/negation_probe.py
from __future__ import annotations
from typing import Literal, Dict
import re

from generation import gen_any
from prompting import system_wrap

Label = Literal["true", "false", "unknown"]

# Simple in-memory cache so we don't re-judge the same sentence repeatedly
_FACT_CHECK_CACHE: Dict[str, Label] = {}


def _clean_one_line(text: str) -> str:
    """
    Take only the first non-empty line and strip bullets / numbering.
    Keeps prompts robust against LLM verbosity.
    """
    if not text:
        return ""

    line = text.split("\n", 1)[0].strip()
    # strip things like "1)", "-", "*", "1." etc
    line = re.sub(r'^\s*[\d\.\)\-\*]+\s*', "", line).strip()
    return line


def build_claim(question: str, head_answer: str, provider: str, model: str) -> str:
    """
    Build a single factual claim that ties the question to the clustered answer.

    Examples:
        Q: Where was Barack Obama born?
        A: United States
        -> Barack Obama was born in the United States.

        Q: Are all toads frogs?
        A: Yes.
        -> All toads are frogs.
    """
    prompt = system_wrap(f"""
You are a careful formatter.

Question:
{question}

Model answer:
{head_answer}

Write ONE short factual sentence that states exactly what the model is claiming.
Requirements:
- A single simple sentence.
- It must be something that can clearly be True or False.
- No explanations, no extra text.

Return only the sentence.
""")

    try:
        raw = gen_any(
            prompt,
            provider=provider,
            model=model,
            k=1,
            max_tokens=60,
        )[0]
    except Exception as e:
        print(f"[NegProbe] claim generation failed: {e!r}")
        raw = ""

    claim = _clean_one_line(raw)

    if not claim:
        # Very generic fallback, still usable for True/False check
        claim = f'"{head_answer}" is a correct answer to the question "{question}".'
    return claim


def build_negated_claim(claim: str, provider: str, model: str) -> str:
    """
    Build the logical opposite of the claim.

    Example:
        Claim: Barack Obama was born in the United States.
        -> Barack Obama was not born in the United States.
    """
    prompt = system_wrap(f"""
Rewrite the following factual claim so that it states the logical opposite meaning.

Claim:
{claim}

Requirements:
- Keep it as ONE short sentence.
- Do not hedge.
- State the direct opposite of the original claim.
- No explanations, no extra text.

Negated claim:
""")

    try:
        raw = gen_any(
            prompt,
            provider=provider,
            model=model,
            k=1,
            max_tokens=60,
        )[0]
    except Exception as e:
        print(f"[NegProbe] negated claim generation failed: {e!r}")
        raw = ""

    negated = _clean_one_line(raw)

    if not negated:
        # Still gives us a structured negation to truth-check
        negated = "It is not true that " + claim
    return negated


def ask_true_false(claim: str, provider: str, model: str) -> Label:
    """
    Ask the model to judge the claim as True, False, or Unknown.

    Output contract is intentionally VERY tight to reduce drift.
    """
    if claim in _FACT_CHECK_CACHE:
        return _FACT_CHECK_CACHE[claim]

    prompt = system_wrap(f"""
You are a precise fact checker.

Statement:
{claim}

Classify this statement with exactly ONE word:
True
False
Unknown
""")

    try:
        raw = gen_any(
            prompt,
            provider=provider,
            model=model,
            k=1,
            max_tokens=3,
        )[0]
    except Exception as e:
        print(f"[NegProbe] fact check failed: {e!r}")
        raw = ""

    ans = raw.strip().lower()

    if ans.startswith("true"):
        label: Label = "true"
    elif ans.startswith("false"):
        label = "false"
    else:
        label = "unknown"

    _FACT_CHECK_CACHE[claim] = label
    return label


def pair_score(orig: Label, neg: Label) -> float:
    """
    Map (orig_label, neg_label) into a risk score in [0, 1].

    Interpretation:
        C  = truth label for claim
        ¬C = truth label for negated claim

        C = True,  ¬C = False   -> Best case, consistent   -> 0.0
        C = False, ¬C = True    -> Answer is wrong        -> 1.0
        C = True,  ¬C = True    -> Complement / paraphrase -> 0.4
        C = False, ¬C = False   -> Confused semantics     -> 0.8
        C = Unknown, ¬C = {T/F} -> Partial hedge          -> 0.5
        C = {T/F}, ¬C = Unknown -> Partial hedge          -> 0.5
        Both Unknown            -> No information         -> 0.6
    """
    # Best case: model believes claim, rejects its negation
    if orig == "true" and neg == "false":
        return 0.0

    # Model believes the negation but not the claim:
    # strong signal that the clustered answer is wrong.
    if orig == "false" and neg == "true":
        return 1.0

    # Both sides true => often paraphrase / complementary framing
    if orig == "true" and neg == "true":
        return 0.4

    # Both sides false => failed to treat them as opposites
    if orig == "false" and neg == "false":
        return 0.8

    # Both unknown => hedgy and low information
    if orig == "unknown" and neg == "unknown":
        return 0.6

    # Mixed unknown with one definite label => medium risk
    return 0.5



def negation_flip_score(
    question: str,
    head_answer: str,
    provider: str,
    model: str,
) -> float:
    """
    Main entry point used by the pipeline.

    Flow:
        1. Build a factual claim from (question, clustered answer).
        2. Build its logical negation.
        3. Ask model: "Is claim True/False/Unknown?"
        4. Ask model: "Is negated claim True/False/Unknown?"
        5. Convert the (orig, neg) pair into a risk score in [0, 1].

    Semantics:
        - Lower score  -> more consistent behavior ("True" then "False").
        - Higher score -> inconsistent or inverted behavior.
    """
    try:
        claim = build_claim(question, head_answer, provider=provider, model=model)
    except Exception as e:
        print(f"[NegProbe] claim build failed: {e!r}")
        claim = f'"{head_answer}" is a correct answer to "{question}".'

    try:
        negated = build_negated_claim(claim, provider=provider, model=model)
    except Exception as e:
        print(f"[NegProbe] negated claim build failed: {e!r}")
        negated = "It is not true that " + claim

    try:
        orig_label = ask_true_false(claim, provider=provider, model=model)
    except Exception as e:
        print(f"[NegProbe] original truth-check failed: {e!r}")
        orig_label = "unknown"

    try:
        neg_label = ask_true_false(negated, provider=provider, model=model)
    except Exception as e:
        print(f"[NegProbe] negated truth-check failed: {e!r}")
        neg_label = "unknown"

    score = pair_score(orig_label, neg_label)
    score = max(0.0, min(1.0, score))

    print(
        f"[TRACE] NegProbe: claim='{claim}' ({orig_label}), "
        f"negated='{negated}' ({neg_label}) -> score={score:.3f}"
    )
    return score
