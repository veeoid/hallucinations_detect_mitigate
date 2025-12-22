# src/evaluate_and_report.py
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from nli_checker import check_nli
import sys

def generate_report(ground_truth_file: str, baseline_results_file: str, pipeline_results_file: str):
    """
    Generates a comprehensive performance report for the hallucination pipeline,
    suitable for a technical paper.
    """
    try:
        gt_df = pd.read_csv(ground_truth_file)
        baseline_df = pd.read_csv(baseline_results_file)
        pipeline_df = pd.read_csv(pipeline_results_file)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please ensure you have run both the baseline and the main benchmark.")
        sys.exit(1)

    # --- Part 1: Initial Data Preparation & Merging ---
    # A hallucination is defined as a contradiction with the ground truth.
    baseline_df['is_hallucination'] = (baseline_df['nli_result'] == 'contradiction').astype(int)
    
    # Merge the data into a single DataFrame for analysis
    report_df = pd.merge(gt_df, pipeline_df, on='question')
    report_df = pd.merge(report_df, baseline_df[['question', 'is_hallucination']], on='question')

    print("\n" + "="*70)
    print("= Hallucination Detection & Mitigation Pipeline: Performance Report")
    print("="*70)

    # --- Part 2: Baseline Performance (Model's Raw Hallucination Rate) ---
    total_questions = len(report_df)
    initial_hallucinations = report_df['is_hallucination'].sum()
    
    print("\n## 1. Baseline Model Performance (Before Pipeline)")
    print("-" * 50)
    print(f"Total Questions Evaluated: {total_questions}")
    print(f"Initial Hallucinations Found: {initial_hallucinations} ({initial_hallucinations/total_questions:.2%})")
    print("-" * 50)

    # --- Part 3: Hallucination Detection Performance ---
    y_true = report_df['is_hallucination']
    # A 'REVIEW' decision means the pipeline predicted a hallucination.
    y_pred = report_df['pipeline_decision'].apply(lambda x: 1 if 'REVIEW' in x else 0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n## 2. Detection Stage Performance")
    print("-" * 50)
    print("This measures the pipeline's ability to correctly identify hallucinations.")
    print(f"  - True Positives (TP):   {tp:3d} (Correctly detected hallucinations)")
    print(f"  - False Positives (FP):  {fp:3d} (Correct answers incorrectly flagged)")
    print(f"  - True Negatives (TN):   {tn:3d} (Correct answers correctly accepted)")
    print(f"  - False Negatives (FN):  {fn:3d} (Missed hallucinations)")
    print("-" * 50)
    print(f"  - Accuracy:  {accuracy:.2%}")
    print(f"  - Precision: {precision:.2%} (Of all answers we flagged, how many were actual hallucinations?)")
    print(f"  - Recall:    {recall:.2%} (Of all actual hallucinations, how many did we catch?)")
    print(f"  - F1-Score:  {f1:.2%}")
    print("-" * 50)

    # --- Part 4: Mitigation & Answer Quality Analysis ---
    print("\n## 3. Mitigation & Final Answer Quality")
    print("-" * 50)
    print("This measures the impact of the pipeline on the final answer's correctness.")
    
    # Analyze the answers for correctness using NLI
    initial_correct = 0
    final_correct = 0
    hallucinations_mitigated = 0
    
    detected_hallucinations_df = report_df[(report_df['is_hallucination'] == 1) & (y_pred == 1)]

    print("\nAnalyzing final answer quality (this may take a moment)...")
    for _, row in tqdm(report_df.iterrows(), total=total_questions, desc="NLI Analysis"):
        reference = str(row['reference'])
        before_answer = str(row['before_answer'])
        after_answer = str(row['pipeline_answer'])

        # Check initial correctness
        if check_nli(reference, before_answer) == 'entailment':
            initial_correct += 1
        
        # Check final correctness
        if check_nli(reference, after_answer) == 'entailment':
            final_correct += 1
        
        # Check if a correctly detected hallucination was successfully mitigated
        if row['is_hallucination'] == 1 and ('REVIEW' in row['pipeline_decision']):
             if check_nli(reference, after_answer) != 'contradiction':
                 hallucinations_mitigated += 1

    print("\n--- Mitigation Results ---")
    if tp > 0:
        print(f"Hallucinations Correctly Detected: {tp}")
        print(f"Successfully Mitigated:            {hallucinations_mitigated} ({hallucinations_mitigated/tp:.2%})")
    else:
        print("No hallucinations were detected, so mitigation was not triggered.")
    
    print("\n--- Overall Answer Quality ---")
    print(f"Correct Answers BEFORE Pipeline: {initial_correct} / {total_questions} ({initial_correct/total_questions:.2%})")
    print(f"Correct Answers AFTER Pipeline:  {final_correct} / {total_questions} ({final_correct/total_questions:.2%})")
    
    net_improvement = final_correct - initial_correct
    print(f"\nNet Improvement in Correct Answers: {net_improvement:+.0f}")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate hallucination detection and mitigation pipeline"
    )

    parser.add_argument(
        "--ground_truth",
        default="data/qa_ground_truth.csv",
        help="Path to ground truth CSV",
    )
    parser.add_argument(
        "--baseline",
        default="data/baseline_benchmark_results.csv",
        help="Path to baseline results CSV",
    )
    parser.add_argument(
        "--input",
        "--pipeline",
        dest="pipeline_results_file",
        default="data/benchmark_results.csv",
        help="Path to pipeline results CSV",
    )

    args = parser.parse_args()

    generate_report(
        ground_truth_file=args.ground_truth,
        baseline_results_file=args.baseline,
        pipeline_results_file=args.pipeline_results_file,
    )
