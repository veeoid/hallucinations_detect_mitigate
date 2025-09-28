# src/runner_closed_book.py
import argparse
import pandas as pd

from generation import gen_ollama
from semantic_entropy import semantic_entropy
from perturbation_consistency import consistency_score
from rules import (
    rules_score,
    question_prior_risk,
    entity_emergence_penalty,
)
from negation_probe import negation_flip_score
from risk import risk_score, decision

CSV_PATH = "data/qa_mini.csv"


def make_paraphrases(q: str) -> list[str]:
    """Generate two short paraphrases for the question (closed-book)."""
    prompts = [
        f"Paraphrase the question concisely, keep meaning identical:\n{q}",
        f"Rephrase the question in other words (no extra info):\n{q}",
    ]
    return [gen_ollama(p, k=1, max_tokens=60)[0] for p in prompts]


def run_one(question: str, model: str, k: int, max_tokens: int):
    print("Question:", question, "\n")

    # 1) Multi-sample generation (k answers)
    samples = gen_ollama(question, model=model, k=k, max_tokens=max_tokens)
    for i, s in enumerate(samples, 1):
        print(f"Sample {i}: {s}\n")

    # 2) Semantic entropy on the k samples
    s_conf, _clusters = semantic_entropy(samples, cos_thresh=0.85)

    # 3) Paraphrase consistency vs. first sample
    paras = make_paraphrases(question)
    para_answers = [gen_ollama(p, model=model, k=1, max_tokens=max_tokens)[0] for p in paras]
    s_consist = consistency_score(samples[0], para_answers)

    # 4) Cheap rule checks on the first sample
    s_rules = rules_score(samples[0])

    # 5) Extra closed-book signals
    q_prior = question_prior_risk(question)
    s_emerge = entity_emergence_penalty(question, samples[0])
    s_flip = negation_flip_score(samples[0])  # tiny yes/no probe with same model

    # 6) Risk fusion + decision
    r = risk_score(s_conf, s_consist, s_rules, q_prior, s_emerge, s_flip)
    dec = decision(r)

    # 7) Print summary
    print(
        f"Semantic Entropy: {s_conf:.3f} | Consistency: {s_consist:.3f} | Rules: {s_rules:.3f}\n"
        f"Q-prior: {q_prior:.3f} | Emergence: {s_emerge:.3f} | NegFlip: {s_flip:.3f}"
    )
    print(f"RISK: {r:.3f}  →  {dec.upper()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0, help="Row index in data/qa_mini.csv")
    ap.add_argument("--model", type=str, default="llama3:instruct", help="Ollama model tag")
    ap.add_argument("--k", type=int, default=3, help="Number of samples for entropy")
    ap.add_argument("--max_tokens", type=int, default=120, help="Max new tokens per sample")
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH)
    if not (0 <= args.idx < len(df)):
        raise IndexError(f"--idx {args.idx} out of range (0..{len(df)-1})")

    q = df.iloc[args.idx]["question"]
    run_one(q, model=args.model, k=args.k, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
