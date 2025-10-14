import pandas as pd
import sys
from tqdm import tqdm

# We need to import functions from our 'src' directory
from runner_closed_book import sample_answers, get_decision_and_metrics
from nli_checker import check_nli

def run_baseline_benchmark(questions_file: str, model: str, provider: str, k: int):
    """
    Runs a baseline benchmark to measure the raw hallucination rate of a model.
    """
    try:
        questions_df = pd.read_csv(questions_file)
    except FileNotFoundError:
        print(f"Error: The questions file was not found at '{questions_file}'")
        sys.exit(1)

    contradictions = 0
    entailments = 0
    neutrals = 0
    total_questions = len(questions_df)

    print(f"\n--- Running Baseline Benchmark for model: {model} ---")
    print(f"Analyzing {total_questions} questions...")

    # Use tqdm for a progress bar
    for _, row in tqdm(questions_df.iterrows(), total=total_questions):
        question = row['question']
        reference_answer = str(row['reference'])

        # 1. Get the model's raw answers
        raw_samples = sample_answers(question, model, provider, k)
        
        # 2. Find the model's most confident "before" answer
        _, metrics = get_decision_and_metrics(raw_samples)
        before_answer = metrics.get('head', 'Unknown')

        # 3. Use NLI to compare the model's answer to the reference
        # We treat "I have no comment" as a neutral case
        if reference_answer == "I have no comment" or before_answer == "Unknown":
            nli_result = 'neutral'
        else:
            nli_result = check_nli(premise=reference_answer, hypothesis=before_answer)

        # 4. Count the results
        if nli_result == 'contradiction':
            contradictions += 1
        elif nli_result == 'entailment':
            entailments += 1
        else:
            neutrals += 1
            
    # --- 5. Report the Final Benchmark ---
    hallucination_rate = contradictions / total_questions
    consistency_rate = entailments / total_questions

    print("\n--- Initial Benchmark Results (Model Raw Performance) ---")
    print("-" * 55)
    print(f"Total Questions: {total_questions}")
    print(f"Model: {model} (via {provider})")
    print("-" * 55)
    print(f"✅ Factual Consistency (Entailment): {entailments} ({consistency_rate:.2%})")
    print(f"❌ Hallucinations (Contradiction):   {contradictions} ({hallucination_rate:.2%})")
    print(f"❔ Neutral / No Comment:             {neutrals}")
    print("-" * 55)
    print("This report shows the baseline performance of the model *before* any intervention from your pipeline.")


if __name__ == "__main__":
    # --- Configuration ---
    # You can easily change the model and provider here
    run_baseline_benchmark(
        questions_file='data/qa_mini.csv',
        model='phi3',
        provider='ollama',
        k=10
    )