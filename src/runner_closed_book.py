# src/runner_closed_book.py
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from typing import List, Tuple, Dict
import os

from generation import gen_any
from risk import classify_risk
from semantic_entropy import cluster_strings_soft
from reality_check import is_fictional
from mitigation import mitigate
from negation_probe import negation_flip_score
from rules import rules_score


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
                pass # File is empty
            for row in reader:
                 if row and row[0].strip():
                    questions.append(row[0].strip())
    return questions


def build_prompt(question: str) -> str:
    return "Just give one standard plausible answer.  Do not explain. Do not be creative.\n" + question

def _is_unknownish(s: str) -> bool:
    """
    This is the rewritten, much stricter version. It only flags clear and
    unambiguous refusal patterns. This fixes the bug from Q38.
    """
    s2 = (s or "").strip().lower()
    # Exact match for simple refusals
    if s2 in {"unknown", "n/a", "na"}:
        return True
    # Starts with a clear refusal phrase
    refusal_starters = (
        "i'm sorry", "i cannot", "i am unable", "i do not have", "there is no",
        "i'm not aware", "i couldn't find"
    )
    if s2.startswith(refusal_starters):
        return True
    return False

def sample_answers(question: str, model: str, provider: str, k: int, max_tok: int = 64) -> List[str]:
    base_prompt = build_prompt(question)
    samples: List[str] = []
    print(f"\n[TRACE] Stage 1: Generating {k} samples...")
    for _ in range(k):
        raw_answer = gen_any(base_prompt, provider=provider, model=model, k=1, max_tokens=max_tok)[0]
        samples.append(raw_answer.strip())
    print(f"[TRACE] Result: {len(samples)} samples generated.")
    return samples


def get_decision_and_metrics(
    question: str,
    values: List[str],
    provider: str,
    model: str
) -> Tuple[str, Dict]:
    print("\n[TRACE] Stage 2: Hallucination Detection...")
    metrics = {}
    
    non_refusal_values = [v for v in values if not _is_unknownish(v) and v]
    if not non_refusal_values:
        print("[TRACE] Finding: All answers were refusals or empty. Low confidence.")
        return "REVIEW", {"head": "Unknown", "confidence": 0.0, "method": "none"}

    labels = cluster_strings_soft(non_refusal_values)
    label_counts = Counter(labels)
    
    # NEW: Calculate diversity penalty
    # This measures how scattered the answers are. 1.0 is perfect consensus. 0.0 is total chaos.
    num_clusters = len(label_counts)
    diversity_penalty = 1.0 - ( (num_clusters - 1) / max(1, len(non_refusal_values) - 1) ) if len(non_refusal_values) > 1 else 1.0


    most_common_label, head_cluster_size = label_counts.most_common(1)[0]
    base_confidence = head_cluster_size / len(non_refusal_values)
    head_answer_index = labels.index(most_common_label)
    head_answer = non_refusal_values[head_answer_index]
    
    metrics["head"] = head_answer
    metrics["method"] = "semantic_clustering"

    print("[TRACE] Action: Applying additional detection signals (negation & rules).")
    negation_score = negation_flip_score(head_answer, provider, model)
    rule_based_risk = rules_score(head_answer, question=question)

    metrics["negation_score"] = round(negation_score, 2)
    metrics["rule_based_risk"] = round(rule_based_risk, 2)
    print(f"[TRACE] Result: Negation score = {negation_score}, Rule risk = {rule_based_risk}")

    # The final confidence is now penalized by the diversity score.
    final_confidence = base_confidence * diversity_penalty * (1 - negation_score) * (1 - rule_based_risk)
    metrics["confidence"] = round(final_confidence, 3)
    
    # Adjusted threshold for the new confidence calculation
    decision = "ACCEPT" if final_confidence > 0.60 else "REVIEW"
    print(f"[TRACE] Result: Base Confidence={base_confidence:.3f}, Diversity Penalty={diversity_penalty:.3f}, Final Confidence is {final_confidence:.3f}. Initial decision is '{decision}'.")
    return decision, metrics

def run_single_trace(args):
    """Runs the full pipeline for a single question for detailed analysis."""
    questions = load_questions(args.csv)
    if not (0 <= args.idx < len(questions)):
        raise IndexError(f"Index {args.idx} is out of range for the {len(questions)} questions in {args.csv}.")

    question = questions[args.idx]

    print("="*50)
    print(f"= Running Pipeline Trace for Question #{args.idx}...")
    # ... rest of the function is the same as the previous correct version ...
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
            model=args.model
        )
    else: # decision == "ACCEPT"
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
                    model=args.model
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
    args = ap.parse_args()

    if args.idx is not None:
        run_single_trace(args)
    else:
        run_full_benchmark(args)

if __name__ == "__main__":
    main()