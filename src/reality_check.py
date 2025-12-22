# src/reality_check.py
from __future__ import annotations
from typing import Literal, Tuple
from generation import gen_any
from prompting import system_wrap

QuestionType = Literal["REAL", "FICTIONAL_OR_MYTH", "UNKNOWN"]
AnswerMode = Literal["REAL_MODE", "FICTIONAL_MODE", "DENY_FICTIONALITY", "UNKNOWN"]


def classify_question_type(
    question: str,
    provider: str,
    model: str,
) -> QuestionType:
    """
    Classify the question as REAL, FICTIONAL_OR_MYTH, or UNKNOWN.
    """
    prompt = system_wrap(f"""
You are a classifier.

Decide what type of domain this question belongs to.

Labels:
- REAL           -> about real world facts, science, history, society, economics, health, etc.
- FICTIONAL_OR_MYTH -> about fictional stories, magic, supernatural beings, myths, folklore, fantasy universes.
- UNKNOWN        -> unclear or mixed.

Question:
{question}

Answer with exactly one label:
REAL
FICTIONAL_OR_MYTH
UNKNOWN
""")

    resp = gen_any(prompt, provider=provider, model=model, k=1, max_tokens=5)[0].strip().upper()

    if "REAL" == resp:
        return "REAL"
    if "FICTIONAL_OR_MYTH" in resp:
        return "FICTIONAL_OR_MYTH"
    return "UNKNOWN"


def classify_answer_mode(
    question: str,
    answer: str,
    provider: str,
    model: str,
) -> AnswerMode:
    """
    Classify how the answer treats reality given the question.
    """
    prompt = system_wrap(f"""
You are a classifier.

Given the question and the model's answer, decide how the answer is treating the situation.

Labels:
- REAL_MODE
    The answer talks about real world facts or denies the existence of unreal things.
    Example: "Vampires are fictional and do not exist in reality."
- FICTIONAL_MODE
    The answer treats fictional or supernatural things as if they are real inside a story world.
    Example: "Vampires avoid sunlight and drink blood."
- DENY_FICTIONALITY
    The answer explicitly says that the thing does not exist in reality.
    Example: "There are no flying carpets in real life."
- UNKNOWN
    The answer is vague, mixed, or cannot be clearly assigned.

Question:
{question}

Answer:
{answer}

Return exactly ONE label from:
REAL_MODE
FICTIONAL_MODE
DENY_FICTIONALITY
UNKNOWN
""")

    resp = gen_any(prompt, provider=provider, model=model, k=1, max_tokens=5)[0].strip().upper()

    if resp == "REAL_MODE":
        return "REAL_MODE"
    if resp == "FICTIONAL_MODE":
        return "FICTIONAL_MODE"
    if resp == "DENY_FICTIONALITY":
        return "DENY_FICTIONALITY"
    return "UNKNOWN"


def reality_mismatch(
    question: str,
    clustered_answer: str,
    provider: str,
    model: str,
) -> Tuple[bool, QuestionType, AnswerMode]:
    """
    Main entry for the pipeline.

    Returns:
        is_mismatch: True if a real world question is answered in fictional mode.
        q_type: classified question type.
        a_mode: classified answer mode.
    """
    q_type = classify_question_type(question, provider=provider, model=model)
    a_mode = classify_answer_mode(question, clustered_answer, provider=provider, model=model)

    mismatch = False

    # If the question is real world, a fictional mode answer is suspicious.
    if q_type == "REAL" and a_mode == "FICTIONAL_MODE":
        mismatch = True

    # If the question is fictional or myth, do not penalize
    # both fictional mode and deny fictional are acceptable.
    # No extra logic needed here for now.

    return mismatch, q_type, a_mode
