# src/logic_plan.py
import json, re
from functools import lru_cache
from typing import Dict, List, Tuple
from generation import gen_any
from breakdown_config import CONFIG

# ---------------- Planner ----------------
PLAN_PROMPT = """
Extract a compact reasoning plan from the question.

Return strict JSON with:
- target: answer type noun like "city", "player", or "country"
- objective: {
    "type": "argmax"|"argmin"|"exists"|"count"|"compare"|"boolean"|"equals"|"procedure"|"definition",
    "metric": string|null,
    "selection": "single"|"topk"|null,
    "k": number|null,
    "slot": string|null,       # e.g., "birth_country" for equals
    "subject": string|null,    # e.g., "Barack Obama" for equals
    "A": string|null,          # for compare
    "B": string|null           # for compare
  }
- constraints: array of items, each EXACTLY one of:
  {"type":"membership","slot":string,"value":string}
  {"type":"predicate_true","slot":string}
  {"type":"predicate_false","slot":string}
  {"type":"count","slot":string,"op":">="|"<="|"=","number":number}
  {"type":"compare","slot":string,"op":"gt"|"ge"|"lt"|"le"|"eq","number":number}
- time_scope: string|null    # e.g., "as of 2012", "currently", "in 2020"

Rules:
- Use "equals" for subject→property value questions (fill subject/slot).
- Use "compare" for A vs B on a metric (fill A, B, metric).
- Use "exists" when asking whether any item satisfies constraints.
- Use "count" when asking how many satisfy constraints.
- Use "topk" when multiple answers are plausible; choose a small k (≤5).
- Do NOT return answers. Plan only.

Question:
"""

def _normalize_plan(plan: Dict) -> Dict:
    if not isinstance(plan, dict):
        return {}
    obj = plan.get("objective") or {}
    t = obj.get("type")
    if t not in {"argmax","argmin","exists","count","compare","boolean","equals","procedure","definition"}:
        obj["type"] = "boolean"
    if obj.get("type") not in {"argmax","argmin"}:
        obj.setdefault("metric", None)
    # equals keys
    if obj.get("type") == "equals":
        obj.setdefault("slot", None)
        obj.setdefault("subject", None)
        obj["selection"] = "single"; obj["k"] = None
    # compare keys
    if obj.get("type") == "compare":
        obj.setdefault("A", None); obj.setdefault("B", None)
        if not obj.get("metric"): obj["metric"] = ""
        obj["selection"] = "single"; obj["k"] = None
    # selection/k
    sel = obj.get("selection")
    if sel not in {"single","topk"}:
        obj["selection"] = "single"; obj["k"] = None
    else:
        if sel == "topk":
            try: k = int(obj.get("k") or CONFIG["set_topk_default"])
            except Exception: k = CONFIG["set_topk_default"]
            obj["k"] = max(1, min(10, k))
        else:
            obj["k"] = None
    # count normalization
    cons = []
    for c in plan.get("constraints", []) or []:
        if isinstance(c, dict) and c.get("type") == "count":
            if "op" not in c: c["op"] = ">="
            if "number" in c:
                try: c["number"] = float(c["number"])
                except Exception: pass
        cons.append(c)
    plan["objective"] = obj
    plan["constraints"] = cons
    plan["time_scope"] = plan.get("time_scope") or None
    return plan

def make_plan(question: str, model: str, provider: str, max_tokens: int = 220) -> Dict:
    prompt = PLAN_PROMPT + question.strip() + "\n\nReturn ONLY JSON on one line."
    raw = gen_any(prompt, model=model, provider=provider, k=1, max_tokens=max_tokens, temps=CONFIG["temp_low1"])[0]
    s, e = raw.find("{"), raw.rfind("}")
    raw_json = raw[s:e+1] if s >= 0 and e >= 0 else "{}"
    try: plan = json.loads(raw_json)
    except Exception: plan = {}
    return _normalize_plan(plan)

# ---------------- Candidates for entity routes ----------------
CANDIDATE_PROMPT = """
Propose up to {maxk} plausible CANDIDATES that satisfy ALL hard constraints in the plan
AND are competitive for the objective (if argmax/argmin).

Hard constraints = membership, predicate_true, predicate_false.

Rules:
- Output plain names only, one per line
- Do NOT number the lines
- No explanations

Plan:
{plan}

Candidates:
"""

def list_candidates(plan: Dict, model: str, provider: str, max_tokens: int = 80) -> List[str]:
    txt = json.dumps(plan, ensure_ascii=False)
    prompt = CANDIDATE_PROMPT.format(maxk=CONFIG["max_candidates"], plan=txt)
    out = gen_any(prompt, model=model, provider=provider, k=1, max_tokens=max_tokens, temps=CONFIG["temp_low1"])[0]
    lines = [re.sub(r"^\s*\d+[\.\)]\s*", "", l).strip() for l in out.splitlines()]
    drop = {"unknown","none","n/a","na","null","-"}
    cands = [l for l in lines if l and l.lower() not in drop and 1 <= len(l.split()) <= 7]
    seen, uniq = set(), []
    for c in cands:
        key = c.lower()
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq[:CONFIG["max_candidates"]]

# ---------------- Semantic concept guard ----------------
_CONCEPT_PROMPT = """Answer strictly Yes or No.
Does this slot describe the concept: "{concept}"?

Slot: "{slot}"

Answer:"""

@lru_cache(maxsize=256)
def slot_is_concept(slot: str, concept: str, model: str, provider: str) -> bool:
    outs = gen_any(_CONCEPT_PROMPT.format(slot=slot, concept=concept),
                   model=model, provider=provider, k=3, max_tokens=3, temps=CONFIG["temps_low3"])
    votes = [o.strip().split()[0].lower() for o in outs]
    return votes.count("yes") >= 2

def forbids_concept(plan: Dict, concept: str, model: str, provider: str) -> bool:
    for c in plan.get("constraints", []) or []:
        if c.get("type") == "predicate_false":
            slot = str(c.get("slot",""))
            if slot and slot_is_concept(slot, concept, model, provider):
                return True
    return False

GUARD_PROMPT = """
You are checking HARD constraints only.

Return strict JSON with keys:
{"ok": true|false, "violations": ["slot", ...]}

Hard constraints are only membership, predicate_true, predicate_false.
A single violation makes ok=false. Do not guess. If uncertain about a slot, leave ok=true.

Plan:
"""

def _property_yesno(entity: str, property_name: str, model: str, provider: str) -> bool:
    p = f'Answer strictly Yes or No.\nIs "{entity}" {property_name}?\nAnswer:'
    outs = gen_any(p, model=model, provider=provider, k=3, max_tokens=3, temps=CONFIG["temps_low3"])
    votes = [o.strip().split()[0].lower() for o in outs]
    return votes.count("yes") >= 2

def guard_candidate(plan: Dict, cand: str, model: str, provider: str, max_tokens: int = 60) -> Tuple[bool, List[str]]:
    txt = json.dumps(plan, ensure_ascii=False)
    prompt = GUARD_PROMPT + txt + "\n\nCandidate: " + str(cand) + "\n\nAnswer with JSON only."
    outs = gen_any(prompt, model=model, provider=provider, k=3, max_tokens=max_tokens, temps=CONFIG["temps_low3"])
    oks, violations = [], []
    for raw in outs:
        s, e = raw.find("{"), raw.rfind("}")
        try: obj = json.loads(raw[s:e+1])
        except Exception: obj = {"ok": True, "violations": []}
        oks.append(bool(obj.get("ok", True)))
        violations.extend(obj.get("violations", []) or [])
    json_ok = all(oks)
    # example concept veto: national capital
    if forbids_concept(plan, concept="national capital", model=model, provider=provider):
        if _property_yesno(cand, "a national capital", model, provider):
            json_ok = False
            violations.append("concept:national_capital")
    return (json_ok), violations

# ---------------- Pairwise + metric compare ----------------
PAIRWISE_PROMPT = """
You will compare two candidates under a plan.

Decision rule:
1) If either candidate violates any HARD constraint (membership, predicate_true, predicate_false), that candidate loses
2) Otherwise choose the one that better satisfies the objective
3) If unclear or equally good, answer Tie
4) If information is insufficient, answer Unknown

Respond exactly one of:
WINNER: A because <10 words>
WINNER: B because <10 words>
WINNER: Tie because <10 words>
WINNER: Unknown because <10 words>

Plan:
{plan}

A = {a}
B = {b}
"""

_METRIC_COMPARE_PROMPT = """Which candidate scores higher on the metric? Answer exactly "A" or "B" or "Unknown".
Metric: {metric}
A = {a}
B = {b}
Answer:"""

def compare_metric(a: str, b: str, metric: str, model: str, provider: str) -> int:
    outs = gen_any(_METRIC_COMPARE_PROMPT.format(metric=metric, a=a, b=b),
                   model=model, provider=provider, k=3, max_tokens=7, temps=CONFIG["temps_low3"])
    votes = [o.strip().split()[0].upper() for o in outs]
    if votes.count("A") >= 2 and votes.count("A") > votes.count("B"): return 1
    if votes.count("B") >= 2 and votes.count("B") > votes.count("A"): return -1
    return 0

def pairwise_vote(plan: Dict, a: str, b: str, model: str, provider: str, trials: int = 3, max_tokens: int = 60) -> Tuple[int,int,int,int,List[str]]:
    txt = json.dumps(plan, ensure_ascii=False)
    prompt = PAIRWISE_PROMPT.format(plan=txt, a=a, b=b)
    outs = gen_any(prompt, model=model, provider=provider, k=trials, max_tokens=max_tokens, temps=CONFIG["temps_low3"])
    va = vb = vu = vt = 0
    reasons: List[str] = []
    for o in outs:
        first = o.strip().splitlines()[0].lower()
        if "winner: a" in first: va += 1
        elif "winner: b" in first: vb += 1
        elif "winner: tie" in first: vt += 1
        else: vu += 1
        reasons.append(o.strip())
    obj = plan.get("objective") or {}
    t, m = obj.get("type"), (obj.get("metric") or "")
    if t in {"argmax","argmin"} and m:
        pref = compare_metric(a, b, m, model, provider)
        if pref == 1 and t == "argmax": va += 1
        elif pref == -1 and t == "argmax": vb += 1
        elif pref == 1 and t == "argmin": vb += 1
        elif pref == -1 and t == "argmin": va += 1
    return va, vb, vu, vt, reasons

# ---------------- Checklist ----------------
CHECKLIST_PROMPT = """
Return JSON array of labels "Yes"|"No"|"Unknown" for EACH constraint in the plan, in order.

Plan:
{plan}

Candidate: {cand}

Output example: ["Yes","Unknown","No"]
"""

def checklist(plan: Dict, cand: str, model: str, provider: str, trials: int = 3, max_tokens: int = 80) -> Tuple[float,float,float,List[List[str]]]:
    txt = json.dumps(plan, ensure_ascii=False)
    prompt = CHECKLIST_PROMPT.format(plan=txt, cand=cand)
    outs = gen_any(prompt, model=model, provider=provider, k=trials, max_tokens=max_tokens, temps=CONFIG["temps_low3"])
    want = len(plan.get("constraints", []))
    arrays, ys, ns, us = [], 0, 0, 0
    for o in outs:
        s, e = o.find("["), o.rfind("]")
        try:
            arr = json.loads(o[s:e+1])
            if want: arr = (arr + ["Unknown"]*want)[:want]
        except Exception:
            arr = ["Unknown"]*max(1, want)
        arrays.append(arr)
        for lab in arr:
            lab = str(lab).strip().lower()
            if lab == "yes": ys += 1
            elif lab == "no": ns += 1
            else: us += 1
    tot = ys + ns + us or 1
    return ys/tot, us/tot, ns/tot, arrays

# ---------------- Equals helpers (multi-probe) ----------------
VALUE_CANDIDATES_PROMPT = """
List up to {maxk} plausible VALUES for the property below.
Output one value per line. No numbering, no explanations.

Subject: {subject}
Property: {slot}
Answer type: {target}

Values:
"""

def list_value_candidates(subject: str, slot: str, target: str, model: str, provider: str) -> List[str]:
    prompt = VALUE_CANDIDATES_PROMPT.format(
        maxk=CONFIG["equals_candidates"], subject=subject, slot=slot, target=target or "value"
    )
    out = gen_any(prompt, model=model, provider=provider, k=1, max_tokens=80, temps=CONFIG["temp_low1"])[0]
    lines = [re.sub(r"^\s*\d+[\.\)]\s*","", l).strip() for l in out.splitlines()]
    drop = {"unknown","none","n/a","na","null","-"}
    vals = [l for l in lines if l and l.lower() not in drop]
    seen, uniq = set(), []
    for v in vals:
        key = v.lower()
        if key not in seen and len(v) <= 60:
            seen.add(key); uniq.append(v)
    return uniq[:CONFIG["equals_candidates"]]

_EQ1 = """Answer strictly one token: Yes or No or Unknown.
Is the {slot} of "{subject}" equal to "{value}"?
Answer:"""

_EQ2 = """Answer strictly one token: Yes or No or Unknown.
Was "{subject}" {slot_alt} "{value}"?
Answer:"""

_EQ_AB = """Answer strictly one token: A or B or Unknown.
Which is correct for the property below?

Subject: {subject}
Property: {slot}
A) {value_a}
B) {value_b}

Answer:"""

def _norm_token(s: str, allowed: List[str]) -> str:
    tok = (s or "").strip().split()[0]
    tok = tok.strip(" .,:;").lower()
    return tok if tok in allowed else "unknown"

def equals_score(subject: str, slot: str, slot_alt: str, value: str, model: str, provider: str) -> float:
    allowed = {"yes","no","unknown"}
    votes = []
    # Probe 1
    outs = gen_any(_EQ1.format(slot=slot, subject=subject, value=value),
                   model=model, provider=provider, k=5, max_tokens=3, temps=CONFIG["temps_low3"])
    votes += [_norm_token(o, allowed) for o in outs]
    # Probe 2
    slot_alt_phrase = slot_alt or ( 'born in' if "birth" in (slot or "").lower() else slot )
    outs2 = gen_any(_EQ2.format(subject=subject, slot_alt=slot_alt_phrase, value=value),
                    model=model, provider=provider, k=5, max_tokens=3, temps=CONFIG["temps_low3"])
    votes += [_norm_token(o, allowed) for o in outs2]
    yes = votes.count("yes"); no = votes.count("no")
    total = max(1, yes + no)  # ignore unknowns in ratio
    return yes / total

def equals_pairwise_choice(subject: str, slot: str, a: str, b: str, model: str, provider: str) -> int:
    outs = gen_any(_EQ_AB.format(subject=subject, slot=slot, value_a=a, value_b=b),
                   model=model, provider=provider, k=5, max_tokens=3, temps=CONFIG["temps_low3"])
    allowed = {"a","b","unknown"}
    toks = [_norm_token(o, allowed) for o in outs]
    if toks.count("a") > toks.count("b") and toks.count("a") >= 3: return +1
    if toks.count("b") > toks.count("a") and toks.count("b") >= 3: return -1
    return 0
