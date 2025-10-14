# src/reality_check.py
from __future__ import annotations
import re
from typing import List
from generation import gen_any

def _extract_key_concepts(text: str) -> List[str]:
    """A simple function to extract noun phrases as key concepts."""
    # This regex is a basic way to find multi-word noun-like phrases.
    # It looks for sequences of capitalized words or sequences of nouns/adjectives.
    concepts = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\b[a-z]+\s+curse\b|\b[a-z]+\s+kiss\b)\b", text)
    
    # Filter for concepts that are more than just a single, common word.
    phrases = [c.strip().lower() for c in concepts if len(c.split()) > 1]
    
    # Add some specific single words if they are highly indicative
    if "fairytale" in text.lower(): phrases.append("fairytale")
    if "prince" in text.lower(): phrases.append("prince")
        
    return list(set(phrases))[:5] # Limit to the top 5 concepts to keep the prompt clean

def is_fictional(question: str, answer: str, provider: str, model: str) -> bool:
    """
    Checks if an answer is likely based on fictional concepts.
    Returns True if the answer is likely fictional, False otherwise.
    """
    concepts = _extract_key_concepts(answer)
    
    # If we can't extract any meaningful concepts, assume it's not fictional.
    if not concepts:
        return False

    concept_list = ", ".join([f'"{c}"' for c in concepts])
    
    prompt = f"""
You are a fact-checker distinguishing reality from fiction.
Consider the main ideas in the following answer. Are these ideas primarily associated with real-world, factual events, or are they primarily associated with fictional stories, myths, and fairytales?

Answer the question with a single word: "Real" or "Fictional".

Question: "{question}"
Answer: "{answer}"
Key Concepts: {concept_list}

Analysis:
"""

    # We poll the model a few times to get a consensus on the check.
    responses = gen_any(prompt, provider=provider, model=model, k=3, max_tokens=5)
    
    # Count the votes
    fictional_votes = sum(1 for r in responses if "fictional" in r.lower())
    
    # If a majority of responses agree it's fictional, return True.
    return fictional_votes >= 2