# src/generation.py
import requests

def gen_ollama(prompt: str,
               model: str = "llama3:instruct",
               k: int = 3,
               temps = (0.6, 0.8, 1.0),
               max_tokens: int = 120) -> list[str]:
    """
    Call Ollama's HTTP API to generate k samples. No shell quoting issues.
    """
    outs = []
    url = "http://localhost:11434/api/generate"
    for i in range(k):
        t = temps[i % len(temps)]
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": t,
                "top_p": 0.95,
                "num_predict": max_tokens
            },
            "stream": False  # get one JSON back
        }
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        outs.append((data.get("response") or "").strip())
    return outs
