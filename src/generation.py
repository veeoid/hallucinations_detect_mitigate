# src/generation.py
from __future__ import annotations
from typing import List
from generation_groq import gen_groq

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
    if provider.lower() != "groq":
        raise RuntimeError(f"Unsupported provider: {provider}")
    return gen_groq(
        prompt=prompt,
        model=model,
        k=k,
        temperature=temperature,
        max_tokens=max_tokens,
        system=system,
    )
