# src/runner_closed_book.py
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from typing import List, Tuple, Dict
import os
import math  # for entropy calc

from generation import gen_any
from risk import classify_risk
from semantic_entropy import cluster_strings_soft
from reality_check import is_fictional
from mitigation import mitigate
from negation_probe import negation_flip_score  # PQPF on the original question
from rules import rules_score
from collections import Counter as C


def load_questions(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        alt = os.path.join("data", os.path.basename(csv_path))
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"CSV not found at {csv_path} or {alt}")

    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = [h.lower().strip() for h in next(reader)]
            q_index = header.index('question')
            for row in reader:
                if row and len(row) > q_index and row[q_index].strip():
                    questions.append(row[q_index].strip())
        except (ValueError, StopIteration, IndexError):
            f.seek(0)
            # Skip header if it exists
            try:
                next(reader)
            except StopIteration:
                pass  # File is empty
            for row in reader:
                if row and row[0].strip():
                    questions.append(row[0].strip())
    return questions


def build_prompt(question: str) -> str:
    return "Just give one standard plausible answer.  Do not explain. Do not be creative.\n" + question


def _is_unknownish(s: str) -> bool:
    """
    Stricter refusal detector: only flags clear, unambiguous refusals.
    """
    s2 = (s or "").strip().lower()
    # Exact match for simple refusals
    if s2 in {"unknown", "n/a", "na"}:
        return True
    # Starts with a clear refusal phrase
    refusal_starters = (
        "i'm sorry", "i cannot", "i am unable", "i do not have", "there is no",
        "i'm not aware", "i couldn't find", "i don't have", "i'm unable", "i cannot provide",
        "i'm afraid", "i do not know", "i'm unable to", "i cannot answer"
    )
    if s2.startswith(refusal_starters):
        return True
    return False


def _normalized_entropy_from_counts(label_counts: Counter, n: int) -> float:
    """
    Shannon entropy over cluster distribution, normalized by log(n).
    Returns 0..1 (higher = more dispersed / less consensus).
    """
    if n <= 1:
        return 0.0
    ps = [cnt / float(n) for cnt in label_counts.values() if cnt > 0]
    H = -sum(p * math.log(p + 1e-12) for p in ps)
    H_norm = H / max(math.log(n + 1e-12), 1e-12)
    return max(0.0, min(1.0, H_norm))


def sample_answers(question: str, model: str, provider: str, k: int, max_tok: int = 64) -> List[str]:
    base_prompt = build_prompt(question)
    samples: List[str] = []
    print(f"\n[TRACE] Stage 1: Generating {k} samples...")
    for _ in range(k):
        raw_answer = gen_any(base_prompt, provider=provider, model=model, k=1, max_tokens=max_tok)[0]
        samples.append(raw_answer.strip())
    print(f"[TRACE] Result: {len(samples)} samples generated.")
    return samples


def _taxonomy_ambiguity(samples: List[str]) -> bool:
    """
    Returns True when multiple samples mention taxonomy/definition cues,
    indicating 'it depends' answers (subset/family/order/definition).
    """
    txt = " ".join((s or "") for s in samples).lower()
    cues = [
        "subset", "sub-class", "subclassification", "classification", "taxonomy", "categor",
        "order anura", "anura", "family bufonidae", "bufonidae", "ranidae", "hylidae",
        "but not all", "type of", "kind of", "form of", "defined as", "definition",
        "sometimes called", "also known as", "may refer to", "varies by context"
    ]
    hits = sum(cue in txt for cue in cues)
    return hits >= 2


def get_decision_and_metrics(
    question: str,
    values: List[str],
    provider: str,
    model: str
) -> Tuple[str, Dict]:
    print("\n[TRACE] Stage 2: Hallucination Detection...")
    metrics: Dict = {}

    # Unknown / refusal handling (computed on all values)
    unknown_ratio = 0.0
    if values:
        unknown_ratio = sum(1 for v in values if _is_unknownish(v)) / float(len(values))
    metrics["unknown_ratio"] = round(unknown_ratio, 3)

    # Filter out refusals for clustering
    non_refusal_values = [v for v in values if not _is_unknownish(v) and v]
    if not non_refusal_values:
        print("[TRACE] Finding: All answers were refusals or empty. Low confidence.")
        return "REVIEW", {
            "head": "Unknown",
            "confidence": 0.0,
            "method": "none",
            "entropy": 0.0,
            "unknown_ratio": metrics["unknown_ratio"]
        }

    # --- Semantic clustering (labels per answer) ---
    # Try stricter threshold to avoid merging distinct names; fall back if API differs
    try:
        labels = cluster_strings_soft(non_refusal_values, similarity_threshold=0.88)
    except TypeError:
        labels = cluster_strings_soft(non_refusal_values)

    label_counts = Counter(labels)

    # Build cluster -> indices map for robust head selection & tracing
    cluster_to_idxs: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        cluster_to_idxs.setdefault(lab, []).append(idx)

    # Diversity penalty (existing heuristic) + floor to avoid crushing borderline splits
    num_clusters = len(label_counts)
    if len(non_refusal_values) > 1:
        diversity_penalty = 1.0 - ((num_clusters - 1) / max(1, len(non_refusal_values) - 1))
    else:
        diversity_penalty = 1.0
    diversity_penalty = max(0.6, diversity_penalty)  # floor
    metrics["diversity_penalty"] = round(diversity_penalty, 3)

    # Normalized entropy (0..1) for reporting
    entropy_norm = _normalized_entropy_from_counts(label_counts, len(non_refusal_values))
    metrics["entropy"] = round(entropy_norm, 3)

    # Largest cluster and a robust representative: majority string *inside* that cluster
    largest_lab, _ = label_counts.most_common(1)[0]
    largest_idxs = cluster_to_idxs[largest_lab]

    
    majority_in_cluster = C(non_refusal_values[i] for i in largest_idxs).most_common(1)[0][0]

    head_answer = majority_in_cluster

    head_cluster_size = len(largest_idxs)
    base_confidence = head_cluster_size / len(non_refusal_values)

    metrics["head"] = head_answer
    metrics["method"] = "semantic_clustering"
    metrics["head_cluster_size"] = head_cluster_size
    metrics["base_confidence"] = round(base_confidence, 3)

    # (Optional) Trace largest cluster members for sanity
    print("[TRACE] Largest cluster members:")
    for i in largest_idxs:
        print("    -", non_refusal_values[i])

    print("[TRACE] Action: Applying additional detection signals (negation & rules).")

    # --- Negation probe: PQPF on the original question ---
    try:
        negation_score = negation_flip_score(question, head_answer,provider, model)
        # 0.0 consistent, 1.0 inconsistent, 0.5 neutral
    except Exception as e:
        print(f"[TRACE] Warning: negation probe failed with error: {e!r}. Using neutral 0.0 penalty.")
        negation_score = 0.0

    # Neutralize negation on taxonomy/definition ambiguity to avoid false "contradiction"
    if negation_score == 1.0 and _taxonomy_ambiguity(non_refusal_values):
        print("[TRACE] Negation neutralized: taxonomy/definition ambiguity detected.")
        negation_score = 0.5

    metrics["negation_score"] = round(negation_score, 3)

    # --- Rules probe (lighten for short entity heads) ---
    short_head = len(head_answer.split()) <= 3
    try:
        rule_based_risk = rules_score(head_answer, question=question)
        if short_head:
            rule_based_risk *= 0.5
    except Exception as e:
        print(f"[TRACE] Warning: rules_score failed with error: {e!r}. Using neutral 0.0 penalty.")
        rule_based_risk = 0.0
    metrics["rule_based_risk"] = round(rule_based_risk, 3)

    print(f"[TRACE] Result: Negation score = {negation_score:.3f}, Rule risk = {rule_based_risk:.3f}")

    # ------------------------------------------------------------------
    # Final confidence: semantic stability with a soft risk penalty
    # ------------------------------------------------------------------
    # Semantic consensus part (what clustering says)
    semantic_conf = base_confidence * diversity_penalty

    # Combine negation and rules into a single risk term
    # Negation is usually more noisy for some questions, so keep weights moderate
    neg_weight = 0.4
    rule_weight = 0.3
    risk_penalty = neg_weight * negation_score + rule_weight * rule_based_risk
    # Clamp to [0,1] for safety
    risk_penalty = max(0.0, min(1.0, risk_penalty))

    final_confidence = semantic_conf * (1.0 - risk_penalty)
    final_confidence = max(0.0, min(1.0, final_confidence))

    # Risk-aware thresholding (stricter for high-risk questions)
    risk_tier = classify_risk(question)
    if risk_tier == "high":
        accept_threshold = 0.70
    elif risk_tier == "medium":
        accept_threshold = 0.60
    else:
        accept_threshold = 0.55

    metrics["risk_tier"] = risk_tier
    metrics["confidence"] = round(final_confidence, 3)
    metrics["accept_threshold"] = round(accept_threshold, 2)

    # Initial decision
    decision = "ACCEPT" if final_confidence > accept_threshold else "REVIEW"

    print(
        f"[TRACE] Result: Base={base_confidence:.3f}, "
        f"Diversity={diversity_penalty:.3f}, Entropy={entropy_norm:.3f}, "
        f"Neg={negation_score:.3f}, Rules={rule_based_risk:.3f} "
        f"-> Final Confidence={final_confidence:.3f} vs Threshold={accept_threshold:.2f} "
        f"-> Initial decision '{decision}'."
    )

    return decision, metrics



def run_single_trace(args):
    """Runs the full pipeline for a single question for detailed analysis."""
    questions = load_questions(args.csv)
    if not (0 <= args.idx < len(questions)):
        raise IndexError(f"Index {args.idx} is out of range for the {len(questions)} questions in {args.csv}.")

    question = questions[args.idx]

    print("="*50)
    print(f"= Running Pipeline Trace for Question #{args.idx}...")
    print(f"\nInput Question: \"{question}\"")
    print(f"Provider: {args.provider}, Model: {args.model}, Samples (k): {args.k}")

    raw_samples = sample_answers(question, args.model, args.provider, args.k)
    print("\n--- Raw Samples Generated ---")
    for i, s in enumerate(raw_samples, 1):
        print(f"  Sample {i}: {s}")
    print("----------------------------")

    decision, metrics = get_decision_and_metrics(question, raw_samples, args.provider, args.model)
    before_answer = metrics.get('head', 'Unknown')

    print("\n[TRACE] Stage 3: Reality Check")
    fictional_override = is_fictional(question, args.provider, args.model)

    if fictional_override:
        print("[TRACE] >>> Reality Check FAILED: Question premise appears to be fictional. Overriding to REVIEW.")
        decision = "REVIEW"
    else:
        print("[TRACE] >>> Reality Check PASSED.")

    print("\n[TRACE] Stage 4: Mitigation")
    if fictional_override:
        print("[TRACE] Action: Question flagged as fictional. Generating a direct refutation.")
        final_text = "The premise of your question is based on fictional concepts."
    elif decision == "REVIEW":
        print(f"[TRACE] Action: Decision is 'REVIEW'. Attempting to mitigate...")
        final_text = mitigate(
                            question=question,
                            base_answer=before_answer,
                            samples=raw_samples,
                            severity=classify_risk(question),
                            provider=args.provider,
                            model=args.model,
                            use_rag=args.rag,
                            rag_mode=args.rag_mode,
                            rag_corpus_dir=args.rag_corpus
                        )
    else:  # decision == "ACCEPT"
        print("[TRACE] Action: Decision is 'ACCEPT'. Passing the head answer through.")
        final_text = before_answer

    print("\n" + "="*50)
    print("= Final Output")
    print("="*50)
    final_decision_text = f"Final Decision: {decision}"
    if fictional_override:
        final_decision_text += " (Overridden by Reality Check)"
    print(final_decision_text)
    print(f"Most Common Answer (Before): \"{before_answer}\"")
    print(f"Returned Answer (After): \"{final_text}\"")
    print(f"Confidence Score: {metrics.get('confidence', 0.0)}")
    print("="*50)


def run_full_benchmark(args):
    """Runs the pipeline over all questions in a CSV and saves the results."""
    questions = load_questions(args.csv)

    if args.limit and args.limit > 0:
        questions = questions[:args.limit]

    print(f"Running full benchmark on {len(questions)} questions...")
    print(f"Results will be saved to: {args.output}")

    with open(args.output, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["question", "before_answer", "pipeline_decision", "pipeline_answer", "confidence"])

        for i, question in enumerate(questions):
            print(f"\n--- Processing Question {i+1}/{len(questions)} ---")
            print(f"Question: {question}")

            raw_samples = sample_answers(question, args.model, args.provider, args.k)
            decision, metrics = get_decision_and_metrics(question, raw_samples, args.provider, args.model)
            before_answer = metrics.get('head', 'Unknown')

            fictional_override = is_fictional(question, args.provider, args.model)
            if fictional_override:
                decision = "REVIEW"

            if fictional_override:
                final_text = "The premise of your question is based on fictional concepts."
            elif decision == "REVIEW":
                final_text = mitigate(
                            question=question,
                            base_answer=before_answer,
                            samples=raw_samples,
                            severity=classify_risk(question),
                            provider=args.provider,
                            model=args.model,
                            use_rag=args.rag,
                            rag_mode=args.rag_mode,
                            rag_corpus_dir=args.rag_corpus
                        )
            else:
                final_text = before_answer

            final_decision_text = f"{decision}{' (Overridden)' if fictional_override else ''}"
            writer.writerow([question, before_answer, final_decision_text, final_text, metrics.get('confidence', 0.0)])
            print(f"Before: '{before_answer}' -> After: '{final_text}' (Decision: {final_decision_text})")

    print(f"\n✅ Benchmark processing complete. Results saved to {args.output}")


def main():
    ap = argparse.ArgumentParser(description="Run hallucination detection and mitigation pipeline.")
    ap.add_argument("--csv", type=str, default="data/qa_ground_truth.csv", help="Input CSV with questions.")
    ap.add_argument("--output", type=str, default="data/benchmark_results.csv", help="Output CSV for benchmark results.")
    ap.add_argument("--provider", type=str, default="ollama", help="LLM provider ('ollama' or 'groq').")
    ap.add_argument("--model", type=str, default="phi3", help="The name of the model to use.")
    ap.add_argument("--k", type=int, default=10, help="Number of samples to generate per question.")
    ap.add_argument("--idx", type=int, help="(Optional) Specify a single question index to trace for debugging.")
    ap.add_argument("--limit", type=int, help="(Optional) Limit the benchmark to the first N questions.")
    ap.add_argument("--rag", action="store_true", help="Enable optional RAG grounding for mitigation.")
    ap.add_argument("--rag_mode", type=str, default="local", choices=["local","live"], help="RAG mode: local or live.")
    ap.add_argument("--rag_corpus", type=str, default="data/corpus", help="Directory for local RAG corpus (if rag_mode=local).")

    args = ap.parse_args()

    if args.idx is not None:
        run_single_trace(args)
    else:
        run_full_benchmark(args)

if __name__ == "__main__":
    main()
