# src/mitigation.py
from typing import List
from prompting import system_wrap
from generation import gen_any
import re

# NEW: live, legal retrieval (Wikipedia + DuckDuckGo Instant Answer)
from retrieval_live import retrieve_live_context  # (src/retrieval_live.py)

# ---------------------------
# Proposer / Skeptic (existing)
# ---------------------------

def _prompt_proposer(q: str, samples: List[str]) -> str:
    unique_samples = sorted(list(set(s.strip().rstrip('.') for s in samples if s and len(s) > 10)))
    if not unique_samples:
        unique_samples = sorted(list(set(s.strip().rstrip('.') for s in samples if s)))
    candidate_list = "\n".join(f"- \"{s}\"" for s in unique_samples)

    return system_wrap(f"""
You are an assistant. Select the single best answer for the user's question from the candidates provided.

Question:
{q}

Candidates:
{candidate_list}

**CRITICAL**: Your response must be ONLY the text of the best candidate answer. Do not add any other words.

Best Answer:
""")

def _prompt_skeptic_final(q: str, proposed_answer: str) -> str:
    """
    Final, hardened skeptic. Intentionally binary to avoid overexplaining.
    """
    return system_wrap(f"""
You are a meticulous fact-checker. Your only goal is to determine if the proposed answer is factually correct.

Question:
{q}

Proposed Answer:
"{proposed_answer}"

YOUR TASK:
Is the proposed answer a factually correct and reliable answer to the question?
- If the answer is completely accurate, respond with the single word: "Yes".
- If the answer is wrong or misleading in any way, respond with the single word: "No".

Your entire response must be only "Yes" or "No".
""")

def _clean_final_answer(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'^\s*[\d\.\-]+\s*', '', text).strip()
    return cleaned

# ---------------------------
# RAG-aware prompts (NEW)
# ---------------------------

def _prompt_rag_answer(q: str, sources_block: str) -> str:
    """
    Strict 'cite-or-abstain' mitigation prompt for grounded answers.
    """
    return system_wrap(f"""
Answer strictly using ONLY the sources below. If the sources do not support a factual answer, reply exactly 'Unknown'.
Use one or two sentences and include inline citations like [1], [2] that refer to the source indices provided.

Question:
{q}

SOURCES:
{sources_block}

Answer:
""")

def _has_citation(s: str) -> bool:
    return bool(re.search(r'\[[0-9]+\]', s))

# ---------------------------
# Public API
# ---------------------------

def mitigate(
    question: str,
    base_answer: str,
    samples: List[str],
    severity: str,
    provider: str,
    model: str,
    use_rag: bool = False,
    rag_mode: str = "local",         # "local" or "live" (we only implement "live" here)
    rag_corpus_dir: str = "data/corpus"  # kept for signature compatibility
) -> str:
    """
    Final mitigation with optional real-time RAG.

    Defaults to your existing closed-book Proposer/Skeptic flow.
    If use_rag and rag_mode == "live":
      - fetch short legal summaries (Wikipedia REST, DuckDuckGo Instant Answer)
      - force a 'cite-or-abstain' answer with inline [1], [2] citations
      - validate presence of citations; if missing, return 'Unknown'
      - still pass through the Skeptic gate for safety
    """

    # Fast-path: low severity → keep head answer
    if severity == 'low':
        return base_answer

    # ---------------------------
    # RAG-LIVE BRANCH (optional)
    # ---------------------------
    if use_rag and rag_mode == "live" and retrieve_live_context is not None:
        print("[INFO] Using RAG-LIVE mitigation flow...")
        ctx = retrieve_live_context(question, top_k=2, timeout=10.0)  # [(src, snippet), ...]
        if ctx:
            # Build numbered source block
            sources_block = "\n\n".join(
                f"[{i+1}] {src}\n{snippet}" for i, (src, snippet) in enumerate(ctx)
            )

            print(f'[TRACE] With retrieved contexts, building RAG prompt:\n{sources_block}\n')
            print(f"[TRACE] Generating RAG answer for question: \"{question}\"")

            rag_prompt = _prompt_rag_answer(question, sources_block)
            rag_answer = gen_any(
                rag_prompt, provider=provider, model=model, k=1, max_tokens=160
            )[0].strip()

            print(f"[TRACE] RAG Answer: \"{rag_answer}\"")

            # Normalize and validate
            if not rag_answer:
                return "Unknown"
            if rag_answer.strip().lower() == "unknown":
                return "Unknown"
            if not _has_citation(rag_answer):
                # No inline citations → do not trust grounded rewrite
                return "Unknown"

            # Skeptic pass (final sanity)
            sk = gen_any(
                _prompt_skeptic_final(question, rag_answer),
                provider=provider, model=model, k=1, max_tokens=3
            )[0].strip().lower()

            if "yes" in sk:
                return _clean_final_answer(rag_answer)
            else:
                return "Unknown"

        # No live context within time budget → abstain (safer than guessing)
        return "Unknown"

    # ------------------------------------
    # CLOSED-BOOK BRANCH (existing logic)
    # ------------------------------------
    proposer_prompt = _prompt_proposer(question, samples)
    proposed_answer = gen_any(
        proposer_prompt,
        provider=provider,
        model=model,
        k=1,
        max_tokens=100
    )[0].strip()

    if not proposed_answer or "unknown" in proposed_answer.lower():
        return "Unknown"

    skeptic_prompt = _prompt_skeptic_final(question, proposed_answer)
    skeptic_analysis = gen_any(
        skeptic_prompt,
        provider=provider,
        model=model,
        k=1,
        max_tokens=3
    )[0].strip().lower()

    if "yes" in skeptic_analysis:
        return _clean_final_answer(proposed_answer)
    else:
        # Keep this trace for debugging; final answer remains 'Unknown'
        print(f"[TRACE] Skeptic rejected the answer. Analysis: \"{skeptic_analysis}\"")
        return "Unknown"
