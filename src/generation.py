# src/generation.py
import os
import requests
from generation_groq import gen_groq

def gen_ollama(prompt: str,
               model: str = "llama3:instruct",
               k: int = 3,
               temps = (0.6, 0.8, 1.0),
               max_tokens: int = 120) -> list[str]:
    """Call Ollama's HTTP API to generate k samples."""
    outs = []
    url = "http://localhost:11434/api/generate"
    for i in range(k):
        t = temps[i % len(temps)]
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {"temperature": t, "top_p": 0.95, "num_predict": max_tokens},
            "stream": False
        }
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        outs.append((data.get("response") or "").strip())
    return outs

def gen_any(prompt: str,
            provider: str = "ollama",
            model: str = "llama3:instruct",
            k: int = 3,
            temps = (0.6, 0.8, 1.0),
            max_tokens: int = 120) -> list[str]:
    """
    Provider-agnostic generator used by runner & mitigation.
    provider: 'ollama' or 'groq'
    """
    provider = (provider or "ollama").lower()
    if provider == "ollama":
        return gen_ollama(prompt, model=model, k=k, temps=temps, max_tokens=max_tokens)
    elif provider == "groq":
        # Ensure API key exists to give a clearer error if not.
        if not (os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")):
            raise RuntimeError("GROQ_API_KEY env var not set")
        return gen_groq(prompt, model=model, k=k, temps=temps, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")
