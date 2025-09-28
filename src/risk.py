def risk_score(s_conf: float, s_consist: float, s_rules: float,
               q_prior: float, s_emerge: float, s_flip: float) -> float:
    """
    Closed-book fusion tuned for 'confident myth' traps.
    Heavier on priors/emergence; entropy/consistency are supportive only.
    """
    r = (
        0.05*s_conf +
        0.05*(1.0 - s_consist) +
        0.20*s_rules +
        0.40*q_prior +
        0.25*s_emerge +
        0.05*s_flip
    )
    # Safety floor: superlative+bio prior AND new named entities should never be 'accept'
    if q_prior >= 0.60 and s_emerge >= 0.50:
        r = max(r, 0.60)
    return min(1.0, r)

def decision(risk: float) -> str:
    if risk < 0.35:  return "accept"
    if risk < 0.60:  return "mitigate-medium"
    return "mitigate-high"
