# src/runner_closed_book.py
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import Counter
from typing import List, Tuple, Dict

from generation import gen_any
from risk import classify_risk, classify_qtype
from semantic_entropy import cluster_strings_soft
from reality_check import is_fictional

# (Helper functions load_questions, build_prompt, etc. are included below but unchanged)
def load_questions(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        alt = os.path.join("data", os.path.basename(csv_path))
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"CSV not found at {csv_path} or {alt}")
    with open(csv_path, "r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if "," in first or ";" in first or "\t" in first:
            try:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows and reader.fieldnames:
                    for cand in ("question", "Question", "prompt", "Prompt", "q"):
                        if cand in reader.fieldnames:
                            return [ (r.get(cand) or "").strip() for r in rows if (r.get(cand) or "").strip() ]
                    first_col = reader.fieldnames[0]
                    return [ (r.get(first_col) or "").strip() for r in rows if (r.get(first_col) or "").strip() ]
            except Exception:
                f.seek(0)
        return [line.strip() for line in f if line.strip()]

def build_prompt(question: str) -> str:
    return "Just give one standard plausible answer.  Do not explain. Do not be creative.\n" + question

def _is_unknownish(s: str) -> bool:
    s2 = (s or "").strip().lower()
    refusal_phrases = {
        "unknown", "n/a", "na",
        "i'm sorry, but i can't comply with that",
        "i'm sorry, but i can't help with that",
        "i cannot answer that question",
        "i am unable to provide an answer"
    }
    return s2 in refusal_phrases or s2.startswith("(unknown)")

def _clean(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u2011", " ").replace("\u2010", " ").replace("\u2013", " ").replace("\u2014", " ").replace("\u2212", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip().rstrip(" .,:¡!?\-–—").lower()
    return s

def sample_answers(question: str, model: str, provider: str, k: int, max_tok: int = 64) -> List[str]:
    print("\n[TRACE] Stage 2: Multi-Answer Sampling")
    print(f"[TRACE] Action: Generating {k} samples from the model...")
    base_prompt = build_prompt(question)
    samples: List[str] = []
    for _ in range(k):
        raw_answer = gen_any(base_prompt, provider=provider, model=model, k=1, max_tokens=max_tok)[0]
        if not _is_unknownish(raw_answer):
            samples.append(raw_answer.strip())
            continue
        retry_prompt = (
            "Please provide the most accurate and neutral statement you can regarding the following question. "
            "If the question's premise is incorrect, please state the correction.\n"
            f"Question: {question}"
        )
        retry_answer = gen_any(retry_prompt, provider=provider, model=model, k=1, max_tokens=max_tok)[0]
        samples.append(retry_answer.strip())
    print(f"[TRACE] Result: {len(samples)} samples generated.")
    return samples

def get_decision_and_metrics(values: List[str]) -> Tuple[str, Dict]:
    print("\n[TRACE] Stage 3: Consensus-Based Detection")
    if not values:
        return "REVIEW", {}

    non_unknown_values = [v for v in values if not _is_unknownish(v)]
    unknown_ratio = 1.0 - (len(non_unknown_values) / len(values))

    if not non_unknown_values:
        print("[TRACE] Finding: All answers were 'Unknown' or refusals. Low confidence.")
        return "REVIEW", {"head": "Unknown", "confidence": 0.0, "unknown_ratio": 1.0, "method": "none"}

    simple_answers = [v for v in non_unknown_values if len(v.split()) <= 3]
    if len(simple_answers) / len(non_unknown_values) > 0.7:
        print("[TRACE] Path: Answers are short. Using 'Simple Consensus' (majority vote).")
        cleaned_simple = [_clean(v) for v in simple_answers]
        counts = Counter(cleaned_simple)
        head_answer, head_count = counts.most_common(1)[0]
        confidence = (head_count / len(non_unknown_values)) * (1 - unknown_ratio)
        decision = "ACCEPT" if confidence > 0.70 else "REVIEW"
        original_head = ""
        for v in non_unknown_values:
            if _clean(v) == head_answer:
                original_head = v
                break
        print(f"[TRACE] Finding: Top answer ('{original_head}') appeared {head_count} times.")
        print(f"[TRACE] Result: Confidence score is {confidence:.3f}. Initial decision is '{decision}'.")
        return decision, {
            "head": original_head, "confidence": round(confidence, 3), 
            "unknown_ratio": round(unknown_ratio, 2), "method": "simple_consensus"
        }
    else:
        print("[TRACE] Path: Answers are descriptive. Using 'Semantic Consensus' (clustering).")
        labels = cluster_strings_soft(non_unknown_values)
        label_counts = Counter(labels)
        most_common_label, head_cluster_size = label_counts.most_common(1)[0]
        largest_cluster_ratio = head_cluster_size / len(non_unknown_values)
        head_answer_index = labels.index(most_common_label)
        head_answer = non_unknown_values[head_answer_index]
        confidence = largest_cluster_ratio * (1 - unknown_ratio)
        decision = "ACCEPT" if confidence >= 0.75 else "REVIEW"
        print(f"[TRACE] Finding: Largest semantic cluster contains {head_cluster_size} answers.")
        print(f"[TRACE] Result: Confidence score is {confidence:.3f}. Initial decision is '{decision}'.")
        return decision, {
            "head": head_answer, "confidence": round(confidence, 3), 
            "unknown_ratio": round(unknown_ratio, 2), "method": "semantic_clustering"
        }

def finalize_answer(
    question: str,
    decision: str,
    metrics: Dict,
) -> str:
    print("\n[TRACE] Stage 5: Mitigation")
    print(f"[TRACE] Action: Finalizing answer based on '{decision}' decision.")
    risk_class = classify_risk(question)
    qtype = classify_qtype(question)
    head_answer = metrics.get("head", "Unknown")

    if decision == "ACCEPT":
        print("[TRACE] Result: Confidence is high. Passing the head answer through.")
        return head_answer

    if qtype in ("factoid", "boolean"):
        print(f"[TRACE] Finding: The question is a '{qtype}' and the decision was 'REVIEW'.")
        print("[TRACE] Result: Prioritizing safety. Abstaining and returning 'Unknown'.")
        return "Unknown"

    if qtype == "opinion":
        print("[TRACE] Finding: The question is an 'opinion' and the decision was 'REVIEW'.")
        print("[TRACE] Result: Returning the most common answer as a best guess.")
        return head_answer
        
    return "Unknown"

# This is the main function to run for the presentation trace.
def main():
    # --- Configuration for the Trace ---
    # Change the idx to trace a different question from your CSV.
    # 0 = "Richest person" (Inconsistent Hallucination)
    # 34 = "Spindle" (Consistent, Fictional Hallucination)
    question_idx_to_trace = 27
    
    provider = "ollama"
    model = "phi3"
    k = 10
    # ------------------------------------

    print("="*50)
    print("= Running Pipeline Trace...")
    print("="*50)

    questions = load_questions("data/qa_mini.csv")
    if question_idx_to_trace < 0 or question_idx_to_trace >= len(questions):
        raise IndexError(f"Index {question_idx_to_trace} is out of range.")

    question = questions[question_idx_to_trace]
    print("\n[TRACE] Stage 1: Question Risk Analysis")
    print(f"[TRACE] Input Question: \"{question}\"")
    risk = classify_risk(question)
    qtype = classify_qtype(question)
    print(f"[TRACE] Result: Classified as risk='{risk}', type='{qtype}'.")

    raw_samples = sample_answers(question, model, provider, k)
    
    print("\n--- Raw Samples Generated ---")
    for i, s in enumerate(raw_samples, 1):
        print(f"  Sample {i}: {s}")
    print("----------------------------")

    decision, metrics = get_decision_and_metrics(raw_samples)

    fictional_override = False
    if decision == "ACCEPT" and metrics.get("head") != "Unknown":
        print("\n[TRACE] Stage 4: Reality Check")
        print("[TRACE] Action: Performing reality check on the high-confidence answer...")
        if is_fictional(question, metrics["head"], provider, model):
            print("[TRACE] >>> Reality Check FAILED: Answer appears to be based on fiction.")
            decision = "REVIEW"
            fictional_override = True
        else:
            print("[TRACE] >>> Reality Check PASSED: Answer appears to be based in reality.")

    final_text = finalize_answer(
        question=question,
        decision=decision,
        metrics=metrics,
    )
    
    print("\n" + "="*50)
    print("= Final Output")
    print("="*50)
    final_decision_text = f"Final Decision: {decision}"
    if fictional_override:
        final_decision_text += " (Overridden by Reality Check)"
    print(final_decision_text)
    print(f"Returned Answer: \"{final_text}\"")
    print("="*50)

if __name__ == "__main__":
    main()