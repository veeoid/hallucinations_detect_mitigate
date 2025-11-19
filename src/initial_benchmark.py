# src/initial_benchmark.py
import sys
import csv
import re

import pandas as pd
from tqdm import tqdm

# We need to import functions from our 'src' directory
from runner_closed_book import sample_answers, get_decision_and_metrics
from nli_checker import check_nli


def normalize_answer(text: str) -> str:
    """
    Clean answers so they do not create ugly multiline CSV cells.

    - Strip leading / trailing space
    - Collapse CRLF to LF
    - Split on newlines, keep first non empty paragraph
    - From that paragraph, keep only the first sentence
    """
    if not text:
        return ""

    # Normalize newlines
    text = text.replace("\r", "\n").strip()

    # Break into non empty paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return ""

    first_para = paragraphs[0]

    # Keep only the first sentence from the first paragraph
    sentences = re.split(r"(?<=[.!?])\s+", first_para)
    if sentences:
        return sentences[0].strip()

    return first_para.strip()


def run_baseline_benchmark(
    questions_file: str,
    output_file: str,
    model: str,
    provider: str,
    k: int,
) -> None:
    """
    Runs a baseline benchmark to measure the raw hallucination rate of a model
    and saves the detailed results to a CSV file.
    """
    try:
        questions_df = pd.read_csv(questions_file)
    except FileNotFoundError:
        print(f"Error: The questions file was not found at '{questions_file}'")
        sys.exit(1)

    print(f"\n--- Running Baseline Benchmark for model: {model} ---")
    print(f"Analyzing {len(questions_df)} questions to establish the baseline...")

    with open(output_file, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["question", "reference_answer", "model_answer", "nli_result"])

        for _, row in tqdm(
            questions_df.iterrows(),
            total=len(questions_df),
            desc="Baseline Analysis",
        ):
            question = str(row["question"]).strip()
            reference_answer_raw = str(row["reference"])

            # Generate model samples and get the "head" answer from clustering
            raw_samples = sample_answers(question, model, provider, k)
            _, metrics = get_decision_and_metrics(
                question, raw_samples, provider, model
            )
            before_answer_raw = metrics.get("head", "Unknown")

            # Normalize both sides so CSV is clean and NLI is less noisy
            reference_answer = normalize_answer(reference_answer_raw)
            model_answer = normalize_answer(before_answer_raw)

            # Use NLI to compare to ground truth
            if reference_answer == "I have no comment" or model_answer == "Unknown":
                nli_result = "neutral"
            else:
                nli_result = check_nli(
                    premise=reference_answer,
                    hypothesis=model_answer,
                )

            writer.writerow([question, reference_answer, model_answer, nli_result])

    print(f"\n✅ Baseline benchmark complete. Detailed results saved to {output_file}")


if __name__ == "__main__":
    # --- Configuration ---
    run_baseline_benchmark(
        questions_file="data/qa_ground_truth.csv",
        output_file="data/baseline_benchmark_results.csv",
        model="phi3",
        provider="ollama",
        k=10,
    )
