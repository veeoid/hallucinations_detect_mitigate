# src/breakdown_stage.py
from itertools import combinations
from typing import Dict, List, Tuple
from collections import Counter

from breakdown_config import CONFIG
from logic_plan import (
    make_plan, list_candidates, guard_candidate,
    pairwise_vote, checklist
)

# ---------- helper: Borda + Condorcet ----------
def _borda_from_pairs(cands: List[str], pair_results: Dict[Tuple[str,str], Tuple[int,int]]) -> Dict[str, float]:
    scores = {c: 0.0 for c in cands}
    for (a,b),(va,vb) in pair_results.items():
        total = max(1, va+vb)
        scores[a] += va/total
        scores[b] += vb/total
    denom = max(1, len(cands)-1)
    for k in scores: scores[k] /= denom
    return scores

def _condorcet(cands: List[str], pair_results: Dict[Tuple[str,str], Tuple[int,int]]):
    def beats(x,y):
        if (x,y) in pair_results: va,vb = pair_results[(x,y)]
        else: vb,va = pair_results[(y,x)]
        return va > vb
    for c in cands:
        if all((c==d) or beats(c,d) for d in cands):
            return c
    return None

# ---------- entity histogram from raw samples (cheap) ----------
def entity_histogram(samples: List[str]) -> Tuple[Counter, float, float]:
    import re
    pat = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
    c = Counter()
    for s in samples or []:
        if not s: continue
        for m in pat.finditer(s):
            tok = m.group(1).strip()
            if tok.lower() in {"yes","no","unknown"}: continue
            c[tok] += 1
    total = sum(c.values()) or 1
    shares = sorted((v/total for v in c.values()), reverse=True)
    top = shares[0] if shares else 0.0
    run = shares[1] if len(shares) > 1 else 0.0
    return c, (1.0 - top), (top - run)

# ---------- main entry ----------
def run_breakdown(
    question: str,
    model: str,
    provider: str,
    raw_samples: List[str],
) -> Dict:
    """
    Returns a dict with:
      plan, candidates, filtered_candidates, pair_results, borda, condorcet,
      features (entity_diversity, entity_margin, pair_agree_rate, pair_unknown_rate,
                borda_top, borda_margin, checklist_sat_rate, checklist_unknown_rate),
      suggestion: "accept_single" | "accept_multi" | "review",
      answers: List[str]
    """
    out: Dict = {
        "plan": None,
        "candidates": [],
        "filtered_candidates": [],
        "pair_results": {},
        "borda": {},
        "condorcet": None,
        "features": {},
        "suggestion": "review",
        "answers": [],
    }

    plan = make_plan(question, model=model, provider=provider)
    out["plan"] = plan

    # shape gate via plan
    obj = (plan.get("objective") or {}).get("type")
    target = str(plan.get("target","")).lower()
    entityish = target in {"city","player","team","country","river","mountain","company","person","book","movie"}
    if obj not in {"argmax","argmin"} or not entityish:
        # still return entity histogram over samples for broader pipeline
        _, e_div, e_margin = entity_histogram(raw_samples)
        out["features"].update({"entity_diversity": e_div, "entity_margin": e_margin})
        return out  # review by default for this stage

    # candidates
    cands = list_candidates(plan, model=model, provider=provider)
    out["candidates"] = cands[:]

    # hard-guard
    valid = []
    for c in cands:
        ok, _ = guard_candidate(plan, c, model=model, provider=provider)
        if ok: valid.append(c)
    if valid: cands = valid
    out["filtered_candidates"] = cands[:]

    if len(cands) < 2:
        return out  # review

    # pairwise tournament
    pair_results: Dict[Tuple[str,str], Tuple[int,int]] = {}
    strong_agreements = 0
    total_pairs = 0
    unknown_votes = 0
    tie_votes = 0
    for a,b in combinations(cands, 2):
        va,vb,vu,vt,_ = pairwise_vote(plan, a, b, model=model, provider=provider, trials=3)
        pair_results[(a,b)] = (va,vb)
        total_pairs += 1
        if va == 3 or vb == 3: strong_agreements += 1
        unknown_votes += vu
        tie_votes += vt
    agree_rate = strong_agreements / max(1,total_pairs)
    out["pair_results"] = {f"{a}__vs__{b}":v for (a,b),v in pair_results.items()}
    borda = _borda_from_pairs(cands, pair_results)
    out["borda"] = borda
    ordered = sorted(borda.items(), key=lambda kv: kv[1], reverse=True)
    cond = _condorcet(cands, pair_results)
    out["condorcet"] = cond

    # checklist on top1 (+top2 optional)
    top1 = ordered[0][0]
    sat1, unk1, _, _ = checklist(plan, top1, model=model, provider=provider)

    # features
    _, e_div, e_margin = entity_histogram(raw_samples)
    out["features"].update({
        "entity_diversity": e_div,
        "entity_margin": e_margin,
        "pair_agree_rate": agree_rate,
        "pair_unknown_rate": unknown_votes / max(1, 3*total_pairs),
        "borda_top": ordered[0][1],
        "borda_margin": ordered[0][1] - (ordered[1][1] if len(ordered)>1 else 0.0),
        "checklist_sat_rate": sat1,
        "checklist_unknown_rate": unk1,
    })

    # decision (multi-answer aware)
    cfg = CONFIG
    top_score = ordered[0][1]
    second = ordered[1][1] if len(ordered)>1 else 0.0
    margin = top_score - second
    co_winners = [c for c,s in ordered if (top_score - s) <= cfg["co_winner_band"]]
    selection = (plan.get("objective") or {}).get("selection") or "single"
    k = int((plan.get("objective") or {}).get("k") or 3)

    if out["features"]["checklist_unknown_rate"] > cfg["unknown_rate_max"]:
        out["suggestion"] = "review"
        return out

    if selection == "single":
        if (agree_rate >= cfg["agree_rate_accept"]
            and margin >= cfg["borda_margin_accept"]
            and sat1 >= cfg["sat_rate_min"]
            and cond is not None):
            out["suggestion"] = "accept_single"
            out["answers"] = [top1]
        else:
            out["suggestion"] = "review"
    else:
        co_winners = co_winners[:k]
        if len(co_winners) >= 2 and agree_rate >= 0.5 and sat1 >= 0.60:
            out["suggestion"] = "accept_multi"
            out["answers"] = co_winners
        else:
            out["suggestion"] = "review"

    return out
