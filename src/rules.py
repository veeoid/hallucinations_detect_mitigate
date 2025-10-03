# src/rules.py
"""
General-purpose, domain-agnostic plausibility checks and priors.
No dataset-specific templates; closed-book friendly.

Exports:
- rules_score(answer) -> float in [0,1]
- question_prior_risk(question) -> float in [0,1]
- entity_emergence_penalty(question, answer) -> float in [0,1]
- nonsense_prior(question) -> float in [0,1]
- time_sensitive_prior(question) -> float in [0,1]
- missing_time_anchor_penalty(question, answer) -> float in [0,1]
"""

import re
from datetime import datetime

# -------------------------------
# Regex primitives
# -------------------------------
YEAR_MIN, YEAR_MAX = 1800, datetime.now().year + 1
YEAR_RX = re.compile(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b")
NUM_RX  = re.compile(r"[-+]?\d+(?:\.\d+)?")

# Proper names: require 2–3 capitalized tokens; ignore common starters
_NAME_RX  = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
_IGNORE_1 = {"A","An","The","In","On","At","For","From","To","I","We","You","He","She","It"}

# Absolute/overconfident claims (universally suspicious)
ABSOLUTE_CLAIMS = [
    r"\b(always|never|impossible|guaranteed|certainly|undeniably)\b",
    r"\b(works|correct|successful)\s+100%\b",
    r"\b(cures\s+all|fixes\s+everything)\b",
]

# Superlatives (question prior; general)
_SUPER_RX = re.compile(
    r"\b(richest|largest|smallest|oldest|youngest|first|fastest|best|most|least|biggest|highest|lowest)\b",
    re.I
)

# Time-sensitive phrasing / volatile topics
TIME_SENSITIVE_Q = re.compile(
    r"\b(latest|current|last|this\s+year|as\s+of\s+now|recent|most\s+recent|today)\b",
    re.I
)
TIME_SENSITIVE_TOPICS = re.compile(
    r"\b(world\s*cup|president|prime\s+minister|ceo|rankings?|champion(ship)?|price|inflation|exchange\s+rate|weather)\b",
    re.I
)

# Gibberish priors
_GIB_REPEAT   = re.compile(r'(.)\1{3,}', re.I)     # any char repeated ≥4 (e.g., aaaa)
_ONLY_LETTERS = re.compile(r'^[a-z]+$', re.I)


# -------------------------------
# Helper extractors
# -------------------------------
def _extract_names(text: str) -> set[str]:
    names = set()
    for m in _NAME_RX.finditer(text or ""):
        first = m.group(1).split()[0]
        if first in _IGNORE_1:
            continue
        names.add(m.group(1).strip().lower())
    return names


# -------------------------------
# Answer-level RULES (domain-agnostic)
# -------------------------------
def year_out_of_range(text: str) -> int:
    return sum(1 for y in map(int, YEAR_RX.findall(text or "")) if y < YEAR_MIN or y > YEAR_MAX)

def contradictory_years(text: str) -> int:
    years = set(map(int, YEAR_RX.findall(text or "")))
    return 1 if len(years) >= 2 else 0

def suspicious_big_numbers(text: str) -> int:
    nums = [n for n in NUM_RX.findall(text or "")]
    ints = []
    for n in nums:
        try:
            val = int(float(n))
            ints.append(val)
        except Exception:
            continue
    bigs = [v for v in ints if abs(v) >= 1_000_000_000]  # ≥ 1B
    return 1 if len(bigs) >= 2 else 0

def absolute_claims(text: str) -> int:
    return sum(1 for p in ABSOLUTE_CLAIMS if re.search(p, text or "", flags=re.I))

def rules_score(answer: str) -> float:
    """
    Map rule triggers to normalized risk in [0,1].
    Each trigger contributes +0.2; capped at 1.0.
    """
    flags = (
        year_out_of_range(answer)
        + contradictory_years(answer)
        + suspicious_big_numbers(answer)
        + absolute_claims(answer)
    )
    return min(1.0, flags * 0.2)


# -------------------------------
# Question priors (general)
# -------------------------------
def question_prior_risk(question: str) -> float:
    """Superlatives and 'winner/only/first' style prompts push models to guess."""
    s = 0.0
    if _SUPER_RX.search(question or ""):
        s += 0.4
    return min(1.0, s)

def nonsense_prior(question: str) -> float:
    """Simple prior that flags gibberish-like queries."""
    q = (question or "").strip()
    q_nospace = q.replace(" ", "")
    s = 0.0
    if _GIB_REPEAT.search(q_nospace):
        s += 0.4
    if _ONLY_LETTERS.match(q_nospace) and len(q_nospace) >= 6 and len(set(q_nospace.lower())) <= 3:
        s += 0.3
    return min(1.0, s)

def time_sensitive_prior(question: str) -> float:
    """
    Boost risk for queries likely to have changed recently (recency phrasing or volatile topics).
    Domain-agnostic: we don't assert the fact, just the query structure is risky for closed-book.
    """
    s = 0.0
    if TIME_SENSITIVE_Q.search(question or ""):
        s += 0.4
    if TIME_SENSITIVE_TOPICS.search(question or ""):
        s += 0.2
    return min(1.0, s)


# -------------------------------
# Entity emergence (fabrication cue)
# -------------------------------
def entity_emergence_penalty(question: str, answer: str) -> float:
    """
    Penalize when answer invents new named entities not present in the question.
    """
    q_names = _extract_names(question or "")
    a_names = _extract_names(answer or "")
    new_names = [n for n in a_names if n not in q_names]
    if not new_names:
        return 0.0
    return min(1.0, 0.5 + 0.25 * (len(new_names) - 1))  # 0.5 first, +0.25 each


# -------------------------------
# Time anchor penalty (answer-level)
# -------------------------------
def missing_time_anchor_penalty(question: str, answer: str) -> float:
    """
    If the question is time-sensitive but the answer lacks an explicit year/date,
    penalize. This is structural, not topical.
    """
    if time_sensitive_prior(question) < 0.4:
        return 0.0
    has_year = bool(YEAR_RX.search(answer or ""))
    return 0.3 if not has_year else 0.0
