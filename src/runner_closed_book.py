# src/runner_closed_book.py
import argparse
import pandas as pd

from generation import gen_any
from semantic_entropy import semantic_entropy
from perturbation_consistency import consistency_score
from rules import (
    rules_score,
    question_prior_risk,
    entity_emergence_penalty,
    nonsense_prior,
    time_sensitive_prior,
    missing_time_anchor_penalty,
)
from negation_probe import negation_flip_score
from risk import risk_score, decision, synthesize_safe_answer
from select_base import choose_base_answer
from slot_extract import slot_agreement, slot_stability_name
from mitigation import mitigate
from prompting import system_wrap   

CSV_PATH = "data/qa_mini.csv"

def make_paraphrases(q: str, model: str, provider: str, max_tokens: int):
    """Generate two short paraphrases for the question (closed-book)."""
    prompts = [
        f"Paraphrase the question concisely, keep meaning identical:\n{q}",
        f"Rephrase the question in other words (no extra info):\n{q}",
    ]
    return [gen_any(p, model=model, provider=provider, k=1, max_tokens=max_tokens)[0] for p in prompts]

def run_one(question: str, model: str, provider: str, k: int, max_tokens: int):
    print("Question:", question, "\n")

    # 1) Multi-sample generation
    samples = gen_any(question, model=model, provider=provider, k=k, max_tokens=max_tokens)
    for i, s in enumerate(samples, 1):
        print(f"Sample {i}: {s}\n")

    # 2) Base answer selection
    base_idx, base_answer, force_mitigate = choose_base_answer(samples, cos_thresh=0.85)
    print(f"Base (cluster medoid) = Sample {base_idx+1}\n")

    # 3) Signals
    s_conf, _ = semantic_entropy(samples, cos_thresh=0.85)
    paras = make_paraphrases(question, model, provider, max_tokens)
    para_answers = [gen_any(p, model=model, provider=provider, k=1, max_tokens=max_tokens)[0] for p in paras]
    s_consist = consistency_score(base_answer, para_answers)
    s_rules   = rules_score(base_answer)
    q_prior   = question_prior_risk(question)
    s_emerge  = entity_emergence_penalty(question, base_answer)
    s_flip    = negation_flip_score(base_answer)
    n_prior   = nonsense_prior(question)
    t_prior   = time_sensitive_prior(question)
    t_anchor  = missing_time_anchor_penalty(question, base_answer)

    # Slot-level
    slot_agree, per_slot = slot_agreement(samples)
    s_stable_name = slot_stability_name(question, base_answer, model=model, provider=provider, trials=5, max_tokens=20)

    # 4) Risk fusion
    risk = risk_score(
        s_conf, s_consist, s_rules, q_prior, s_emerge,
        s_flip, n_prior, slot_agree, s_stable_name, t_prior, t_anchor
    )
    dec = decision(risk)

    # 5) Print scores
    print(
        f"Entropy:{s_conf:.3f} | Consist:{s_consist:.3f} | Rules:{s_rules:.3f} | "
        f"Qprior:{q_prior:.3f} | Emergence:{s_emerge:.3f} | NegFlip:{s_flip:.3f} | "
        f"Nonsense:{n_prior:.3f} | Tprior:{t_prior:.3f} | Tanchor:{t_anchor:.3f}\n"
        f"SlotAgree:{slot_agree:.3f} (names:{per_slot['names']:.3f}, years:{per_slot['years']:.3f}, nums:{per_slot['numbers']:.3f}) | "
        f"SlotStableName:{s_stable_name:.3f}"
    )
    print(f"RISK: {risk:.3f}  →  {dec.upper()}")

    returned = base_answer
    if dec.startswith("mitigate"):
        sev = "medium" if dec == "mitigate-medium" else "high"
        returned = mitigate(
            question=question,
            base_answer=base_answer,
            samples=samples,
            severity=sev,
            provider=provider,
            model=model
        )

    print("\nBase decision:", dec.upper())
    print("Returned answer:", returned)
    return base_answer, dec, risk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0, help="Row index in data/qa_mini.csv")
    ap.add_argument("--model", type=str, default="llama3:instruct", help="Model tag")
    ap.add_argument("--provider", type=str, default="ollama", help="Provider: ollama | groq")
    ap.add_argument("--k", type=int, default=20, help="Number of samples")
    ap.add_argument("--max_tokens", type=int, default=120, help="Max tokens per sample")
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH)
    if not (0 <= args.idx < len(df)):
        raise IndexError(f"--idx {args.idx} out of range (0..{len(df)-1})")

    q = df.iloc[args.idx]["question"]
    run_one(q, model=args.model, provider=args.provider, k=args.k, max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
