from typing import List
import requests

def gen_ollama(prompt: str,
               model: str = "llama3:instruct",
               k: int = 3,
               temps = 1.0,
               max_tokens: int = 120) -> List[str]:
    url = "http://localhost:11434/api/generate"
    outs = []
    for i in range(k):
        t = temps[i % len(temps)]
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {"temperature": t, "top_p": 0.95, "num_predict": max_tokens},
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        outs.append((data.get("response") or "").strip())
    return outs

def gen(prompt: str, provider: str = "ollama", **kwargs) -> List[str]:
    provider = (provider or "ollama").lower()
    if provider == "ollama":
        return gen_ollama(prompt, **kwargs)
    elif provider == "groq":
        from generation_groq import gen_groq
        return gen_groq(prompt, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
