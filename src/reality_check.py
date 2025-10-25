# src/reality_check.py
from __future__ import annotations
from typing import List
from generation import gen_any

def is_fictional(question: str, provider: str, model: str) -> bool:
    """
    This new version abandons keyword matching in favor of a more robust,
    prompt-based classification of the question's domain.
    """
    
    # This prompt asks the LLM to categorize the question, which is a more reliable
    # method than simple keyword searching.
    prompt = f"""
You are an expert in categorizing questions. Your task is to determine if the following question belongs to a domain of fiction, mythology, or the supernatural.

Categorize the question into one of the following:
- "Fictional": If the question is about concepts from stories, folklore, myths, magic, or supernatural beings (e.g., vampires, witches, fairy tales).
- "Factual": If the question is about real-world topics like science, history, culture, health, or general knowledge.

Question: "{question}"

Category (Fictional or Factual):
"""

    responses = gen_any(prompt, provider=provider, model=model, k=3, max_tokens=5)
    
    # We check for a consensus among the responses.
    fictional_votes = sum(1 for r in responses if "fictional" in r.lower())
    
    # The premise is only considered fictional if there is a clear majority vote.
    return fictional_votes >= 2