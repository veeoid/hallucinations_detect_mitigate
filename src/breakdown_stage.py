# src/breakdown_stage.py
from itertools import combinations
from typing import Dict, List, Tuple
from collections import Counter

from generation import gen_any
from breakdown_config import CONFIG
from logic_plan import (
    make_plan, list_candidates, guard_candidate,
    pairwise_vote, checklist,
    list_value_candidates, equals_score, equals_pairwise_choice
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

# ---------- Boolean consensus ----------
_BOOL_PROMPT = """Answer strictly one token: Yes or No or Unknown.

Question:
{q}

Answer:"""

def _boolean_votes(q: str, model: str, provider: str, n: int) -> List[str]:
    outs = gen_any(
        _BOOL_PROMPT.format(q=q),
        model=model,
        provider=provider,
        k=n,
        max_tokens=3,
        temps=CONFIG["temps_low3"],
    )
    votes = []
    for o in outs:
        if not o or not o.strip():
            votes.append("unknown")
            continue
        tok = (o or "").strip().split()[0].lower()
        if tok not in {"yes", "no", "unknown"}: tok = "unknown"
        votes.append(tok)
    return votes

def _negate_statement(q: str) -> str:
    s = q.strip()
    lower = s.lower()
    if lower.startswith("can "):   return s[:4] + "not " + s[4:]
    if lower.startswith("is "):    return s[:3] + "not " + s[3:]
    if lower.startswith("are "):   return s[:4] + "not " + s[4:]
    if lower.startswith("do "):    return s[:3] + "not " + s[3:]
    if lower.startswith("does "):  return s[:5] + "not " + s[5:]
    if lower.startswith("did "):   return s[:4] + "not " + s[4:]
    if lower.endswith("?"):        return s[:-1] + " (not)?"
    return "NOT: " + s

# ---------- main entry ----------
def run_breakdown(
    question: str,
    model: str,
    provider: str,
    raw_samples: List[str],
) -> Dict:
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

    obj = (plan.get("objective") or {}).get("type")
    time_scope = plan.get("time_scope")

    # Temporal prior: closed-book sanity only (we still proceed)
    if CONFIG["require_time_for_volatile"] and (obj in {"equals","argmax","argmin","compare","count","exists"}):
        if CONFIG["time_sensitive_prior"] >= 0.30 and not time_scope:
            # Proceed, but many routes may end in REVIEW if uncertainty remains.
            pass

    # ----- BOOLEAN -----
    if obj == "boolean":
        n = CONFIG["bool_samples"]
        votes = _boolean_votes(question, model, provider, n)
        yes = votes.count("yes"); no = votes.count("no"); unk = votes.count("unknown")
        yes_rate = yes / n; no_rate = no / n; unk_rate = unk / n

        q_neg = _negate_statement(question)
        votes_neg = _boolean_votes(q_neg, model, provider, max(4, n//2))
        yes_neg = votes_neg.count("yes"); no_neg = votes_neg.count("no")
        flipped = (no_neg >= yes_neg) if yes_rate >= no_rate else (yes_neg >= no_neg)

        out["features"].update({
            "bool_yes_rate": yes_rate,
            "bool_no_rate": no_rate,
            "bool_unknown_rate": unk_rate,
            "bool_flip_check": 1.0 if flipped else 0.0,
        })

        maj = "yes" if yes_rate >= no_rate else "no"
        maj_rate = max(yes_rate, no_rate)
        if maj_rate >= CONFIG["bool_accept_rate"] and unk_rate <= CONFIG["bool_unknown_max"] and flipped:
            out["suggestion"] = "accept_single"
            out["answers"] = ["Yes" if maj == "yes" else "No"]
        else:
            out["suggestion"] = "review"
        return out

    # ----- EQUALS (subject → property value) -----
    if obj == "equals":
        subj = (plan["objective"] or {}).get("subject") or ""
        slot = (plan["objective"] or {}).get("slot") or ""
        target = (plan.get("target") or "")
        if not subj or not slot:
            _, e_div, e_margin = entity_histogram(raw_samples)
            out["features"].update({"entity_diversity": e_div, "entity_margin": e_margin})
            return out

        values = list_value_candidates(subj, slot, target, model, provider)
        out["candidates"] = values[:]
        if not values:
            return out

        # 1) per-value scores from multi-probe yes/no
        slot_alt = "born in" if "birth" in slot.lower() else None
        per_scores = []
        for v in values:
            s = equals_score(subj, slot, slot_alt, v, model, provider)
            per_scores.append((v, s))

        # 2) head-to-head A/B boosts
        bonus = {v: 0.0 for v,_ in per_scores}
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                a, b = values[i], values[j]
                ch = equals_pairwise_choice(subj, slot, a, b, model, provider)
                if ch == +1:
                    bonus[a] += 0.05; bonus[b] -= 0.05
                elif ch == -1:
                    bonus[b] += 0.05; bonus[a] -= 0.05

        # 3) combine and rank
        combined = []
        for v, s in per_scores:
            combined.append((v, max(0.0, min(1.0, s + bonus[v]))))
        combined.sort(key=lambda x: x[1], reverse=True)

        out["borda"] = {v: s for v, s in combined}
        top = combined[0][1]; runner = combined[1][1] if len(combined) > 1 else 0.0
        margin = top - runner

        out["features"].update({
            "equals_top_score": top,
            "equals_margin": margin,
            "equals_candidate_count": float(len(combined)),
        })

        if top >= CONFIG["equals_accept_rate"] and margin >= CONFIG["equals_margin_min"]:
            out["suggestion"] = "accept_single"
            out["answers"] = [combined[0][0]]
        else:
            out["suggestion"] = "review"
        return out

    # ----- COMPARE (A vs B on metric) -----
    if obj == "compare":
        A = (plan["objective"] or {}).get("A") or ""
        B = (plan["objective"] or {}).get("B") or ""
        metric = (plan["objective"] or {}).get("metric") or ""
        if not A or not B:
            return out
        wins_A = wins_B = unk = 0
        for _ in range(CONFIG["compare_repeats"]):
            prompt = f'Which candidate is higher on the metric? Answer exactly "A" or "B" or "Unknown".\nMetric: {metric}\nA = {A}\nB = {B}\nAnswer:'
            o = gen_any(prompt, model=model, provider=provider, k=1, max_tokens=7, temps=CONFIG["temp_low1"])[0]
            tok = (o or "").strip().split()[0].upper()
            if tok == "A": wins_A += 1
            elif tok == "B": wins_B += 1
            else: unk += 1
        total = max(1, wins_A + wins_B)
        maj_rate = max(wins_A, wins_B) / total
        out["features"].update({
            "compare_A_votes": float(wins_A),
            "compare_B_votes": float(wins_B),
            "compare_unknown_votes": float(unk),
            "compare_majority_rate": maj_rate,
        })
        if maj_rate >= CONFIG["compare_majority"]:
            out["suggestion"] = "accept_single"
            out["answers"] = [A if wins_A > wins_B else B]
        else:
            out["suggestion"] = "review"
        return out

    # ----- EXISTS (does anything satisfy constraints?) -----
    if obj == "exists":
        cands = list_candidates(plan, model=model, provider=provider)
        out["candidates"] = cands[:]
        if not cands:
            return out
        verified = 0
        unknown_accum = 0.0
        filtered = []
        for c in cands:
            ok, _ = guard_candidate(plan, c, model=model, provider=provider)
            if not ok: continue
            sat, unk, _, _ = checklist(plan, c, model=model, provider=provider)
            filtered.append((c, sat, unk))
            if sat > 0 and unk < 0.5:
                verified += 1
                unknown_accum += unk
        out["filtered_candidates"] = [c for c,_,_ in filtered]
        avg_unknown = (unknown_accum / max(1, verified)) if verified else 1.0
        out["features"].update({
            "exists_verified": float(verified),
            "exists_avg_unknown": avg_unknown,
        })
        if verified >= CONFIG["exists_pass_min"] and avg_unknown <= CONFIG["exists_unknown_max"]:
            out["suggestion"] = "accept_single"
            out["answers"] = ["Yes"]
        else:
            out["suggestion"] = "review"
        return out

    # ----- COUNT (how many satisfy constraints?) -----
    if obj == "count":
        cands = list_candidates(plan, model=model, provider=provider)
        out["candidates"] = cands[:]
        if not cands:
            return out
        true_count = 0
        unknowns = 0
        tested = 0
        for c in cands:
            ok, _ = guard_candidate(plan, c, model=model, provider=provider)
            if not ok: continue
            sat, unk, _, _ = checklist(plan, c, model=model, provider=provider)
            tested += 1
            if sat > 0 and unk < 0.5:
                true_count += 1
            elif unk >= 0.5:
                unknowns += 1
        unk_frac = unknowns / max(1, tested)
        out["features"].update({
            "count_verified": float(true_count),
            "count_tested": float(tested),
            "count_unknown_frac": unk_frac,
        })
        if tested >= CONFIG["count_min_tested"] and unk_frac <= CONFIG["count_unknown_max"]:
            out["suggestion"] = "accept_single"
            out["answers"] = [str(true_count)]
        else:
            out["suggestion"] = "review"
        return out

    # ----- SET RETRIEVAL (list all that satisfy X) -----
    selection = (plan.get("objective") or {}).get("selection") or ""
    if selection == "topk":
        k = int((plan.get("objective") or {}).get("k") or  CONFIG["set_topk_default"])
        cands = list_candidates(plan, model=model, provider=provider)
        out["candidates"] = cands[:]
        if not cands:
            return out
        verified = []
        unknown_sum = 0.0
        for c in cands:
            ok, _ = guard_candidate(plan, c, model=model, provider=provider)
            if not ok: continue
            sat, unk, _, _ = checklist(plan, c, model=model, provider=provider)
            if sat > 0 and unk < 0.6:
                verified.append((c, 1.0 - unk))
                unknown_sum += unk
        verified.sort(key=lambda kv: kv[1], reverse=True)
        kept = [c for c,_ in verified[:k]]
        avg_unknown = (unknown_sum / max(1, len(verified))) if verified else 1.0
        out["filtered_candidates"] = [c for c,_ in verified]
        out["features"].update({
            "set_verified": float(len(verified)),
            "set_avg_unknown": avg_unknown,
        })
        if len(verified) >= CONFIG["set_min_verified"] and avg_unknown <= CONFIG["set_unknown_max"]:
            out["suggestion"] = "accept_multi" if len(kept) > 1 else "accept_single"
            out["answers"] = kept if kept else []
        else:
            out["suggestion"] = "review"
        return out

    # ----- ARGMAX / ARGMIN over entity-like targets -----
    target = str(plan.get("target","")).lower()
    entityish = target in {"city","player","team","country","river","mountain","company","person","book","movie","language","university"}
    if obj in {"argmax","argmin"} and entityish:
        cands = list_candidates(plan, model=model, provider=provider)
        out["candidates"] = cands[:]
        valid = []
        for c in cands:
            ok, _ = guard_candidate(plan, c, model=model, provider=provider)
            if ok: valid.append(c)
        if valid: cands = valid
        out["filtered_candidates"] = cands[:]
        if len(cands) < 2:
            return out
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

        top1 = ordered[0][0]
        sat1, unk1, _, _ = checklist(plan, top1, model=model, provider=provider)

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

        top_score = ordered[0][1]
        second = ordered[1][1] if len(ordered)>1 else 0.0
        margin = top_score - second
        co_winners = [c for c,s in ordered if (top_score - s) <= CONFIG["co_winner_band"]]
        selection = (plan.get("objective") or {}).get("selection") or "single"
        k = int((plan.get("objective") or {}).get("k") or 3)

        if out["features"]["checklist_unknown_rate"] > CONFIG["unknown_rate_max"]:
            out["suggestion"] = "review"
            return out

        if selection == "single":
            if (agree_rate >= CONFIG["agree_rate_accept"]
                and margin >= CONFIG["borda_margin_accept"]
                and sat1 >= CONFIG["sat_rate_min"]
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

    # ----- PROCEDURE / DEFINITION (safe abstain) -----
    if obj in {"procedure","definition"}:
        _, e_div, e_margin = entity_histogram(raw_samples)
        out["features"].update({"entity_diversity": e_div, "entity_margin": e_margin})
        out["suggestion"] = "review"
        return out

    # ----- Default: cheap features + review -----
    _, e_div, e_margin = entity_histogram(raw_samples)
    out["features"].update({"entity_diversity": e_div, "entity_margin": e_margin})
    out["suggestion"] = "review"
    return out
