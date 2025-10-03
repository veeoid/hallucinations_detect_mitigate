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

def decision(risk: float) -> str:
    if risk < 0.35:  return "accept"
    if risk < 0.60:  return "mitigate-medium"
    return "mitigate-high"
