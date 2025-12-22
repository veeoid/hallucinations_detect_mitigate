# src/initial_benchmark.py
import pandas as pd
import sys
import csv
from tqdm import tqdm

# We need to import functions from our 'src' directory
from runner_closed_book import sample_answers, get_decision_and_metrics
from nli_checker import check_nli

def run_baseline_benchmark(questions_file: str, output_file: str, model: str, provider: str, k: int):
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

    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["question", "reference_answer", "model_answer", "nli_result"])

        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Baseline Analysis"):
            question = row['question']
            reference_answer = str(row['reference'])

            # Get the model's raw answers and determine its most common response
            raw_samples = sample_answers(question, model, provider, k)
            _, metrics = get_decision_and_metrics(question, raw_samples, provider, model)
            before_answer = metrics.get('head', 'Unknown')

            # Use NLI to compare the model's raw answer to the ground truth
            if reference_answer == "I have no comment" or before_answer == "Unknown":
                nli_result = 'neutral'
            else:
                nli_result = check_nli(premise=reference_answer, hypothesis=before_answer)

            writer.writerow([question, reference_answer, before_answer, nli_result])

    print(f"\n✅ Baseline benchmark complete. Detailed results saved to {output_file}")


if __name__ == "__main__":
    # --- Configuration ---
    run_baseline_benchmark(
        questions_file='data/qa_ground_truth.csv',
        output_file='data/baseline_benchmark_results.csv',
        model='phi3',
        provider='ollama',
        k=10
    )