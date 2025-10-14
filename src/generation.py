# src/generation.py
from __future__ import annotations
from typing import List
from generation_groq import gen_groq
from generation_ollama import gen_ollama # Import the new Ollama function

def gen_any(
    prompt: str,
    provider: str,
    model: str,
    k: int = 3,
    max_tokens: int = 96,
    temperature: float = 0.6,
    system: str | None = None,
    **_,
) -> List[str]:
    provider_lower = provider.lower()
    
    if provider_lower == "groq":
        return gen_groq(
            prompt=prompt,
            model=model,
            k=k,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
    # --- ADDED OLLAMA SUPPORT ---
    elif provider_lower == "ollama":
        return gen_ollama(
            prompt=prompt,
            model=model,
            k=k,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
    # ----------------------------
    else:
        raise RuntimeError(f"Unsupported provider: {provider}. Use 'groq' or 'ollama'.")