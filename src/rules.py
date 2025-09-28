# src/rules.py

import re
from datetime import datetime

YEAR_MIN, YEAR_MAX = 1800, datetime.now().year + 1
YEAR_RX = re.compile(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b")
NUM_RX  = re.compile(r"[-+]?\d+(?:\.\d+)?")

RISKY_TEMPLATES = [
    r"didn['’]t\s+finish\s+(high\s+school|school)",
    r"did\s+not\s+finish\s+(high\s+school|school)",
    r"didn['’]t\s+graduate\s+(high\s+school|school)",
    r"did\s+not\s+graduate\s+(high\s+school|school)",
    r"never\s+(completed|finished)\s+(high\s+school|school)",
    r"left\s+(high\s+school|school)\s+early",
    r"dropped?\s+out\s+of\s+(high\s+school|school)",
    r"works\s+100%\s+of\s+the\s+time",
    r"cure[s]?\s+all",
]


def year_out_of_range(text: str) -> int:
    return sum(1 for y in map(int, YEAR_RX.findall(text)) if y < YEAR_MIN or y > YEAR_MAX)

def contradictory_years(text: str) -> int:
    years = set(map(int, YEAR_RX.findall(text)))
    return 1 if len(years) >= 2 else 0

def suspicious_big_numbers(text: str) -> int:
    nums = [n for n in NUM_RX.findall(text)]
    bigs = [n for n in nums if len(str(int(float(n)))) >= 9]
    return 1 if len(bigs) >= 2 else 0

def risky_templates(text: str) -> int:
    return sum(1 for p in RISKY_TEMPLATES if re.search(p, text, flags=re.I))

def rules_score(answer: str) -> float:
    flags = (
        year_out_of_range(answer)
        + contradictory_years(answer)
        + suspicious_big_numbers(answer)
        + risky_templates(answer)
    )
    return min(1.0, flags * 0.2)  # 0..1

# --- extra closed-book signals (add at bottom or near imports) ---
import re

# Superlatives & education-claim patterns: classic myth traps
_SUPER_RX = re.compile(r"\b(biggest|largest|richest|oldest|youngest|first|fastest|most)\b", re.I)
_BIO_EDU_RX = re.compile(
    r"(didn['’]t|did not)\s+finish\s+(high\s+school|school)|drop(?:ped)?\s+out\s+of\s+(high\s+school|school)",
    re.I
)

def question_prior_risk(question: str) -> float:
    s = 0.0
    if _SUPER_RX.search(question): s += 0.4
    if _BIO_EDU_RX.search(question): s += 0.3
    return min(1.0, s)

# Light proper-noun heuristic (no spaCy): flags new PERSON/ORG-like names in answers
_PROPN_SEQ = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

def entity_emergence_penalty(question: str, answer: str) -> float:
    q_names = set(m.group(1).lower() for m in _PROPN_SEQ.finditer(question))
    a_names = set(m.group(1).lower() for m in _PROPN_SEQ.finditer(answer))
    new_names = [n for n in a_names if n not in q_names]
    if not new_names:
        return 0.0
    return min(1.0, 0.5 + 0.25*len(new_names))  # 0.5 for first new name, +0.25 each (cap 1.0)
