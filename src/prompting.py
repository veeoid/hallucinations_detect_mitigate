import re
from datetime import datetime

TIME_SENSITIVE_Q = re.compile(
    r"\b(latest|current|last|this\s+year|as\s+of\s+now|recent|most\s+recent|today)\b", re.I
)
TIME_SENSITIVE_TOPICS = re.compile(
    r"\b(world\s*cup|president|prime\s+minister|ceo|rankings?|champion(ship)?|price|inflation|exchange\s+rate|weather)\b",
    re.I,
)

def is_time_sensitive(q: str) -> bool:
    return bool(TIME_SENSITIVE_Q.search(q) or TIME_SENSITIVE_TOPICS.search(q))

def truncate_question(q: str, max_chars: int = 300) -> str:
    q = q.strip()
    if len(q) <= max_chars:
        return q
    cut = q[:max_chars]
    last_space = cut.rfind(" ")
    return (cut if last_space < 0 else cut[:last_space]).strip() + " …"

# src/prompting.py

def system_wrap(body: str) -> str:
    """
    Lightweight 'system' wrapper for single-string prompts.
    Works with both chat and completion backends since we pass a flat prompt.
    """
    return (
        "You are a careful, closed-book answer repairer. "
        "Follow the rules precisely; be concise and avoid speculation.\n\n"
        "=== TASK ===\n"
        f"{body}\n"
        "=== END TASK ===\n"
    )


def make_prompt(question: str) -> str:
    q_trim = truncate_question(question)
    ts = is_time_sensitive(q_trim)

    rules = [
        "Answer in ONE sentence. Be precise.",
        "Do NOT introduce new names/dates unless present in the question.",
        "If unsure, output exactly: Unknown.",
        "Do NOT add extra details, lists, or sources.",
    ]
    if ts:
        rules.insert(1, "Include the explicit YEAR for any time-volatile fact.")
        rules.append("If timeframe is ambiguous, ask one clarifying question instead of guessing.")

    rules_txt = "\n- ".join([""] + rules)
    prompt = (
        f"You are a careful assistant. Follow these rules strictly:{rules_txt}\n\n"
        f"Question:\n{q_trim}\n\nRespond:"
    )
    return prompt
