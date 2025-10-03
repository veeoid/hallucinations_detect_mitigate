# src/runner_closed_book.py
import argparse
import pandas as pd

from prompting import make_prompt
from semantic_entropy import semantic_entropy
from perturbation_consistency import consistency_score
from negation_probe import negation_flip_score
from select_base import choose_base_answer
from slot_extract import slot_agreement, slot_stability_name
from rules import (
    rules_score,
    question_prior_risk,
    entity_emergence_penalty,
    nonsense_prior,
    time_sensitive_prior,
    missing_time_anchor_penalty,
)
from risk import risk_score, decision

# unified generator (Ollama by default; Groq optional)
from generation import gen as gen_any


CSV_PATH = "data/qa_mini.csv"


def make_paraphrases(q: str, model: str, max_tokens: int, provider: str) -> list[str]:
    """Generate two short paraphrases for the raw question, then wrap with strict prompt."""
    prompts = [
        f"Paraphrase the question concisely, keep meaning identical:\n{q}",
        f"Rephrase the question in other words (no extra info):\n{q}",
    ]
    # create paraphrase texts first
    paras = [gen_any(p, provider=provider, model=model, k=1, max_tokens=60)[0] for p in prompts]
    # then wrap each paraphrase with our strict prompt
    return [make_prompt(pq) for pq in paras]


def run_one(question: str, model: str, k: int, max_tokens: int,
           base_provider: str, probe_provider: str):
    print("Question:", question, "\n")

    # 1) Build strict prompt to reduce rambling & encourage time anchors
    steered_prompt = make_prompt(question)

    # 2) Multi-sample generation (k answers) from BASE provider
    samples = gen_any(
        steered_prompt, provider=base_provider, model=model, k=k, max_tokens=max_tokens
    )
    for i, s in enumerate(samples, 1):
        print(f"Sample {i}: {s}\n")

    # 3) Choose consensus-medoid as base
    base_idx, base_answer = choose_base_answer(samples, cos_thresh=0.85)
    print(f"Base (cluster medoid) = Sample {base_idx+1}\n")

    # 4) Signals
    # 4a) k-sample stability
    s_conf, _ = semantic_entropy(samples, cos_thresh=0.85)
    slot_agree, per_slot = slot_agreement(samples)

    # 4b) Paraphrase consistency (answers from PROBE provider)
    para_prompts = make_paraphrases(question, model=model, max_tokens=max_tokens, provider=probe_provider)
    para_answers = [
        gen_any(p, provider=probe_provider, model=model, k=1, max_tokens=max_tokens)[0]
        for p in para_prompts
    ]
    s_consist = consistency_score(base_answer, para_answers)

    # 4c) Answer-only plausibility & emergence
    s_rules  = rules_score(base_answer)
    s_emerge = entity_emergence_penalty(question, base_answer)

    # 4d) Question priors
    q_prior  = question_prior_risk(question)
    n_prior  = nonsense_prior(question)
    t_prior  = time_sensitive_prior(question)
    t_anchor = missing_time_anchor_penalty(question, base_answer)

    # 4e) Simple contradiction probe (can also be run with probe provider if implemented that way)
    s_flip   = negation_flip_score(base_answer)

    # 4f) Slot stability on main name (use PROBE provider to decorrelate)
    s_stable_name = slot_stability_name(
    question, base_answer, model=model, provider=probe_provider, trials=5, max_tokens=20
    )

    # 5) Fuse to risk & decide
    risk = risk_score(
        s_conf, s_consist, s_rules, q_prior, s_emerge, s_flip,
        n_prior, slot_agree, s_stable_name, t_prior, t_anchor
    )
    dec = decision(risk)

    # 6) Report
    print(
        f"Entropy:{s_conf:.3f} | Consist:{s_consist:.3f} | Rules:{s_rules:.3f} | "
        f"Qprior:{q_prior:.3f} | Emergence:{s_emerge:.3f} | NegFlip:{s_flip:.3f} | "
        f"Nonsense:{n_prior:.3f} | Tprior:{t_prior:.3f} | Tanchor:{t_anchor:.3f}\n"
        f"SlotAgree:{slot_agree:.3f} (names:{per_slot['names']:.3f}, years:{per_slot['years']:.3f}, nums:{per_slot['numbers']:.3f}) | "
        f"SlotStableName:{s_stable_name:.3f}"
    )
    print(f"RISK: {risk:.3f}  →  {dec.upper()}\n")

    return base_answer, dec, risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0, help="Row index in data/qa_mini.csv")
    ap.add_argument("--model", type=str, default="llama3:instruct", help="Model tag (Ollama or Groq)")
    ap.add_argument("--k", type=int, default=20, help="Number of samples for entropy/slot agreement")
    ap.add_argument("--max_tokens", type=int, default=120, help="Max new tokens per sample")
    ap.add_argument("--question", type=str, default="", help="Override question text (bypass CSV)")
    ap.add_argument("--provider", type=str, default="ollama", help="Base provider: ollama|groq")
    ap.add_argument("--probe_provider", type=str, default="", help="Probe provider override: ollama|groq (defaults to provider)")
    args = ap.parse_args()

    # Resolve providers
    base_provider = (args.provider or "ollama").lower()
    probe_provider = (args.probe_provider or base_provider).lower()

    # Load question
    if args.question.strip():
        q = args.question.strip()
    else:
        df = pd.read_csv(CSV_PATH)
        if not (0 <= args.idx < len(df)):
            raise IndexError(f"--idx {args.idx} out of range (0..{len(df)-1})")
        q = str(df.iloc[args.idx]["question"])

    base_answer, dec, risk = run_one(
        q, model=args.model, k=args.k, max_tokens=args.max_tokens,
        base_provider=base_provider, probe_provider=probe_provider
    )
    print("Base answer:", base_answer)


if __name__ == "__main__":
    main()
