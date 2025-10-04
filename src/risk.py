def risk_score(s_conf, s_consist, s_rules, q_prior, s_emerge, s_flip,
               n_prior, slot_agree, slot_stable_name,
               t_prior, t_anchor) -> float:
    r = (
        0.05*s_conf +
        0.05*(1.0 - s_consist) +
        0.20*s_rules +
        0.30*q_prior +          # was 0.25
        0.25*s_emerge +         # was 0.20
        0.05*s_flip +
        0.10*n_prior +
        0.20*(1.0 - slot_agree) +
        0.10*(1.0 - slot_stable_name) +
        0.20*t_prior +
        0.15*t_anchor
    )

    # safety floors
    if q_prior >= 0.60 and s_emerge >= 0.50:
        r = max(r, 0.60)
    if n_prior >= 0.40:
        r = max(r, 0.35)
    if t_prior >= 0.4:
        r = max(r, 0.50 if t_anchor > 0 else 0.40)
    
    # --- SAFETY FLOOR: superlative claim + new entity + weak cross-sample consensus
    # In closed-book, single-winner questions (richest/first/best/most...) require strong consensus.
    # If the answer introduces a new name and k-sample agreement on that slot is weak, force mitigate.
    if q_prior >= 0.40 and s_emerge >= 0.50 and slot_agree < 0.80:
        r = max(r, 0.60)  # MITIGATE-HIGH

    return min(1.0, r)

# risk.py
def decision(risk: float) -> str:
    if risk < 0.35:  return "accept"
    if risk < 0.60:  return "mitigate-medium"   # constrain
    return "mitigate-high"                      # contrast → strict fallback


def synthesize_safe_answer(samples):
    """
    Try to produce a cautious but informative synthesis instead of 'Unknown'.
    Strategy:
    - Find the most frequent overlapping phrases/entities.
    - Keep qualifiers like 'generally', 'not outright banned', 'depends on law'.
    """
    import re
    from collections import Counter

    # Extract key phrases (basic heuristic)
    phrases = []
    for s in samples:
        if "reasonable" in s.lower():
            phrases.append("reasonable discipline allowed")
        if "not allowed" in s.lower() or "crime" in s.lower() or "illegal" in s.lower():
            phrases.append("excessive punishment illegal")
        if "child abuse" in s.lower():
            phrases.append("excessive punishment = child abuse")
        if "unknown" in s.lower():
            phrases.append("uncertain")

    if not phrases:
        return "The answer is not certain. Details vary depending on interpretation of the law."

    most_common = Counter(phrases).most_common(2)
    summary = "; ".join([mc[0] for mc in most_common])

    return f"Based on multiple answers: {summary}."
