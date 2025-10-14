# src/evaluate_benchmark.py
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from nli_checker import check_nli

def evaluate(ground_truth_file: str, results_file: str):
    """
    Evaluates the pipeline's detection and its impact on final answer quality.
    """
    try:
        ground_truth_df = pd.read_csv(ground_truth_file)
        results_df = pd.read_csv(results_file)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        sys.exit(1)

    ground_truth_df.columns = ground_truth_df.columns.str.strip()
    results_df.columns = results_df.columns.str.strip()

    merged_df = pd.merge(ground_truth_df, results_df, on='question')

    # --- Part 1: Evaluate the DETECTION performance ---
    print("\nRunning NLI analysis to label ground truth...")
    is_hallucination_labels = []
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Analyzing for contradictions"):
        premise = str(row['reference'])
        hypothesis = str(row['before_answer'])
        nli_result = check_nli(premise, hypothesis)
        is_hallucination = 1 if nli_result == 'contradiction' else 0
        is_hallucination_labels.append(is_hallucination)

    y_true = pd.Series(is_hallucination_labels)
    y_pred = (merged_df['pipeline_decision'] == 'REVIEW').astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- Part 1: Hallucination DETECTION Benchmark ---")
    print(f"Total Questions Evaluated: {len(merged_df)}")
    print("-" * 50)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print("-" * 50)

    # --- Part 2: Evaluate the ANSWER QUALITY ---
    print("\nRunning NLI analysis to check answer quality...")
    before_matches = 0
    after_matches = 0
    
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Analyzing for entailment"):
        premise = str(row['reference'])
        
        # Check if the 'before' answer entails the reference
        before_hypothesis = str(row['before_answer'])
        if check_nli(premise, before_hypothesis) == 'entailment':
            before_matches += 1
            
        # Check if the 'after' answer entails the reference
        after_hypothesis = str(row['pipeline_answer'])
        # We also treat "Unknown" as a "correct" answer if the reference is "I have no comment"
        if after_hypothesis == "Unknown" and premise == "I have no comment":
            after_matches += 1
        elif check_nli(premise, after_hypothesis) == 'entailment':
            after_matches += 1

    print("\n--- Part 2: Answer Quality (Before vs. After) ---")
    print("This measures how many final answers are factually consistent with the reference.")
    print("-" * 50)
    print(f"Correct Answers BEFORE Pipeline: {before_matches} / {len(merged_df)} ({before_matches/len(merged_df):.2%})")
    print(f"Correct Answers AFTER Pipeline:  {after_matches} / {len(merged_df)} ({after_matches/len(merged_df):.2%})")
    print("-" * 50)

    # --- Visualization ---
    confusion_matrix_path = '../nli_confusion_matrix.png'
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Not Hallucination (ACCEPT)', 'Predicted Hallucination (REVIEW)'],
                yticklabels=['Actual Not Hallucination', 'Actual Hallucination'])
    plt.title('NLI-Based Hallucination Detection Confusion Matrix')
    plt.ylabel('Ground Truth (from NLI)')
    plt.xlabel('Pipeline Prediction')
    plt.savefig(confusion_matrix_path)
    print(f"\nConfusion matrix saved to {confusion_matrix_path}")

if __name__ == "__main__":
    evaluate('./data/qa_mini.csv', './data/benchmark_results.csv')