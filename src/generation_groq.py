# src/generation_groq.py
from __future__ import annotations
import os
from typing import List
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # read .env so GROQ_API_KEY is available

# One global client; simple & fast
_api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
_client = Groq(api_key=_api_key) if _api_key else Groq()

def gen_groq(
    prompt: str,
    model: str = "openai/gpt-oss-20b",
    k: int = 3,
    temperature: float = 0.6,
    max_tokens: int = 96,
    system: str | None = None,
) -> List[str]:
    """
    Minimal Groq generator (non-stream). Returns k short strings.
    - Uses max_completion_tokens (Groq's param).
    - No 'UNKNOWN' instruction; just returns whatever the model says.
    - If the API returns empty, we return 'Unknown' once as a safety net.
    """
    outs: List[str] = []
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for _ in range(k):
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=messages + [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=1000,  # important for Groq
                top_p=1,
                stream=False,  # keep it simple & reliable
            )
            text = (resp.choices[0].message.content or "").strip()
            outs.append(text or "Unknown")
        except Exception:
            outs.append("Unknown")
    return outs
