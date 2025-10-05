# src/runner_closed_book.py
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import Counter
from typing import List, Tuple

from generation import gen_any                  # your existing generator
from risk import classify_risk, classify_qtype  # simple heuristics

# ---------------------------
# CSV loading (robust)
# ---------------------------

def load_questions(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        alt = os.path.join("data", os.path.basename(csv_path))
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"CSV not found at {csv_path} or {alt}")

    with open(csv_path, "r", encoding="utf-8") as f:
        # Try DictReader first; fall back to raw if header missing
        first = f.readline()
        f.seek(0)
        if "," in first or ";" in first or "\t" in first:
            try:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows and reader.fieldnames:
                    # prefer a question-like column
                    for cand in ("question", "Question", "prompt", "Prompt", "q"):
                        if cand in reader.fieldnames:
                            return [ (r.get(cand) or "").strip() for r in rows if (r.get(cand) or "").strip() ]
                    # else first column
                    first_col = reader.fieldnames[0]
                    return [ (r.get(first_col) or "").strip() for r in rows if (r.get(first_col) or "").strip() ]
            except Exception:
                f.seek(0)
        # Raw lines
        return [line.strip() for line in f if line.strip()]

# ---------------------------
# Sampling
# ---------------------------

def build_prompt(question: str) -> str:
    # Print only the first line later to avoid clutter in "Samples" section.
    return "Just give one standard plausible answer.  Do not explain. Do not be creative.\n" + question

def _is_unknownish(s: str) -> bool:
    s2 = (s or "").strip().lower()
    return s2 in {"", "unknown", "(unknown)", "n/a", "na", "(no-empty-guard) best guess requested"}

def _clean(s: str) -> str:
    s = (s or "").strip()
    # Normalize weird hyphens to plain spaces so theme matching works
    s = s.replace("\u2011", " ").replace("\u2010", " ").replace("\u2013", " ").replace("\u2014", " ").replace("\u2212", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip().rstrip(" .,:;!?\-–—").lower()
    return s

def sample_answers(question: str, model: str, provider: str, k: int, max_tok: int = 64) -> List[str]:
    base_prompt = build_prompt(question)
    outs = gen_any(base_prompt, provider=provider, model=model, k=k, max_tokens=max_tok)

    fixed: List[str] = []
    for s in outs:
        cs = _clean(s)
        if cs == "" or _is_unknownish(cs):
            reask = gen_any(base_prompt, provider=provider, model=model, k=1, max_tokens=max_tok)[0]
            cs2 = _clean(reask)
            fixed.append(cs2 or cs or "unknown")
        else:
            fixed.append(cs)
    return fixed

# ---------------------------
# Metrics
# ---------------------------

def shannon_entropy(values: List[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    n = float(sum(counts.values()))
    if n <= 0:
        return 0.0
    probs = [c / n for c in counts.values()]
    H = -sum(p * math.log(p) for p in probs if p > 0)
    K = len(counts)
    return float(H / math.log(K)) if K > 1 else 0.0

def head_string(values: List[str]) -> Tuple[str, float]:
    if not values:
        return ("", 0.0)
    counts = Counter(values)
    # If there are other answers, ignore "unknown" as head
    if len(counts) > 1 and "unknown" in counts:
        del counts["unknown"]
    head, cnt = counts.most_common(1)[0]
    ratio = cnt / max(1, len(values))
    return head, ratio

def diversity(values: List[str]) -> float:
    if not values:
        return 0.0
    return len(set(values)) / float(len(values))

def unknown_ratio(values: List[str]) -> float:
    if not values:
        return 0.0
    unk = sum(1 for v in values if _is_unknownish(v))
    return unk / float(len(values))

# ---------------------------
# Opinion consensus
# ---------------------------

THEME_KEYWORDS = {
    "confidence": {"confidence", "confident"},
    "practicality": {"practical", "practicality", "convenience", "convenient", "low maintenance", "low-maintenance"},
    "a personal style choice": {"style", "personal style", "fashion", "modern", "bold", "look", "aesthetic"},
    "independence": {"independence", "independent", "individuality", "self-expression", "self expression"},
}

def _theme_hits(text: str) -> set[str]:
    t = text.lower()
    # normalize odd hyphens/dashes
    t = t.replace("\u2011", " ").replace("\u2010", " ").replace("\u2013", " ").replace("\u2014", " ").replace("\u2212", " ")
    hits = set()
    for theme, kws in THEME_KEYWORDS.items():
        if any(k in t for k in kws):
            hits.add(theme)
    return hits

def opinion_consensus_summary(samples: List[str]) -> Tuple[str, float]:
    per_sample_hit = [bool(_theme_hits(s)) for s in samples]
    coverage = sum(per_sample_hit) / max(1, len(samples))
    theme_counts = {}
    for s in samples:
        for th in _theme_hits(s):
            theme_counts[th] = theme_counts.get(th, 0) + 1
    if not theme_counts:
        return ("", coverage)
    top = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    tops = [t for t, _ in top[:3]]
    if len(tops) == 1:
        sent = f"It usually reflects {tops[0]}."
    elif len(tops) == 2:
        sent = f"It usually reflects {tops[0]} and {tops[1]}."
    else:
        sent = f"It usually reflects {tops[0]}, {tops[1]}, or {tops[2]}."
    return (sent, coverage)

# ---------------------------
# Finalizer (risk + qtype)
# ---------------------------

def finalize_answer(
    question: str,
    decision: str,          # "ACCEPT" or "REVIEW"
    head_answer: str,
    head_ratio: float,
    ent: float,
    unk_ratio: float,
    samples: List[str],
    debug: bool = False,
) -> str:
    risk = classify_risk(question)
    qtype = classify_qtype(question)

    if debug:
        print(f"\n[DEBUG] risk={risk} qtype={qtype} head_ratio={head_ratio:.2f} ent={ent:.3f} unk={unk_ratio:.2f}")

    if decision == "ACCEPT":
        return head_answer

    # REVIEW cases
    if qtype in ("factoid", "boolean"):
        # Be conservative on factual/boolean unless cluster was confident
        if debug:
            print("[DEBUG] REVIEW + (factoid/boolean): returning Unknown")
        return "Unknown"

    # opinion path (low-stakes default)
    summary, cov = opinion_consensus_summary(samples)
    if debug:
        print(f"[DEBUG] opinion coverage={cov:.2f} summary={summary!r}")

    if cov >= 0.60 and summary:
        return summary

    # still give a safe opinion fallback (avoid 'Unknown' for subjective)
    return "It’s a personal style choice and varies by individual."

# ---------------------------
# Decision rule
# ---------------------------

def decide(values: List[str]) -> Tuple[str, dict]:
    head, hratio = head_string(values)
    ent = shannon_entropy(values)
    div = diversity(values)
    unk = unknown_ratio(values)

    accept = (hratio >= 0.60) and (unk <= 0.20) and (ent <= 0.70)

    decision = "ACCEPT" if accept else "REVIEW"
    metrics = {
        "head": head,
        "head_ratio": round(hratio, 2),
        "entropy": round(ent, 3),
        "diversity": round(div, 3),
        "unknown_ratio": round(unk, 2),
    }
    return decision, metrics

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--csv", type=str, default="data/qa_mini.csv")
    ap.add_argument("--provider", type=str, default="groq")
    ap.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    questions = load_questions(args.csv)
    if args.idx < 0 or args.idx >= len(questions):
        raise IndexError(f"idx {args.idx} out of range (0..{len(questions)-1})")

    question = questions[args.idx]
    print("\nQuestion")
    print("--------")
    print(question)

    # Show only the first prompt line to keep logs clean
    print("\nPrompt: Just give one plausible answer. Do not explain.")

    # 1) sample
    raw_samples = sample_answers(question, args.model, args.provider, args.k)
    print("\nSamples")
    print("-------")
    for i, s in enumerate(raw_samples, 1):
        print(f"Sample {i}: {s}")

    # 2) clean & decide
    cleaned = [_clean(x) for x in raw_samples]
    decision, metrics = decide(cleaned)

    # 3) reporting
    print("\nCluster & Stability")
    print("-------------------")
    print(f"Head = {metrics['head']}")
    print(f"Entropy:{metrics['entropy']} | Diversity:{metrics['diversity']}")

    print("\nDecision")
    print("--------")
    print(f"Base decision: {decision}")
    print(f"HeadRatio:{metrics['head_ratio']:.2f} | UnknownRatio:{metrics['unknown_ratio']:.2f} | Entropy:{metrics['entropy']:.3f}")

    # 4) finalize with risk-aware policy
    final_text = finalize_answer(
        question=question,
        decision=decision,
        head_answer=metrics["head"],
        head_ratio=metrics["head_ratio"],
        ent=metrics["entropy"],
        unk_ratio=metrics["unknown_ratio"],
        samples=cleaned,
        debug=args.debug,
    )
    print("\nReturned answer:", final_text)

if __name__ == "__main__":
    main()
