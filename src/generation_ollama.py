# src/generation_ollama.py
from __future__ import annotations
import ollama
from typing import List

# A single, persistent client is efficient
_client = ollama.Client()

def gen_ollama(
    prompt: str,
    model: str = "llama3", # A common default for Ollama
    k: int = 3,
    temperature: float = 0.6,
    max_tokens: int = 96, # Note: Ollama doesn't have a strict max_tokens param like OpenAI
    system: str | None = None,
) -> List[str]:
    """
    Ollama generator (non-stream). Returns k short strings.
    """
    outs: List[str] = []
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    
    messages.append({"role": "user", "content": prompt})

    for i in range(k):
        try:
            # Ollama's generate function is simpler for single prompts
            resp = _client.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens, # num_predict is the closest equivalent to max_tokens
                }
            )
            text = (resp['message']['content'] or "").strip()
            print(f"[Ollama Response {i+1}/{k}]: '{text}'")
            outs.append(text or "Unknown")

        except Exception as e:
            print(f"[Ollama Error {i+1}/{k}]: {e}")
            outs.append("Unknown")
            # If there's an error (e.g., server not running), stop trying.
            break 
            
    # Pad the rest of the results with "Unknown" if we broke early
    while len(outs) < k:
        outs.append("Unknown")

    return outs