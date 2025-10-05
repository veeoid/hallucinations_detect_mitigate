# src/risk.py
from __future__ import annotations
import re
from typing import Literal

Risk = Literal["low", "medium", "high"]
QType = Literal["factoid", "boolean", "opinion"]

# High-stakes heuristics (tight)
_HIGH_RX = re.compile(
    r"\b("
    r"diagnos|symptom|treat|dosage|side effect|contraindicat|"
    r"legal|law|statute|liabilit|immigration|asylum|attorney|"
    r"invest|stock|crypto|financial advice|tax|loan|mortgage|"
    r"election|vote|polling place|ballot"
    r")\b",
    re.I,
)

# Medium-stakes heuristics (factual & time-varying)
_MEDIUM_RX = re.compile(
    r"\b("
    r"history|date|population|capital|currency|distance|"
    r"definition|meaning of|"
    r"born|birth|who is|where is|"
    r"most popular|largest|biggest|percentage|rank|times|year|"
    r"president|ceo|net worth"
    r")\b",
    re.I,
)

def classify_risk(question: str) -> Risk:
    q = (question or "").strip()
    if _HIGH_RX.search(q):
        return "high"
    if _MEDIUM_RX.search(q):
        return "medium"
    return "low"

def classify_qtype(question: str) -> QType:
    q = (question or "").strip().lower()
    # opiniony/open-ended
    if "what does it mean" in q or "what is the meaning" in q or "why " in q:
        return "opinion"
    if re.search(r"\b(meaning|mean)\b", q) and ("what" in q or "why" in q):
        return "opinion"
    # direct factoid
    if re.match(r"^(who|what|when|where|which|how many|how much)\b", q):
        return "factoid"
    # boolean yes/no style
    if re.match(r"^(is|are|can|do|does|did|will|should|could|would|may|must)\b", q):
        return "boolean"
    # default safe
    return "opinion"
