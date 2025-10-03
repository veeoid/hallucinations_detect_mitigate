# src/generation_groq.py
import os, requests
from dotenv import load_dotenv
load_dotenv()  # will read .env in your project root


# NOTE: Groq uses an OpenAI-compatible Chat Completions API.
# Model IDs can vary; examples include "llama3-8b-8192" or "mixtral-8x7b-32768".
# Set GROQ_API_KEY in your environment.

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# src/generation_groq.py
from typing import List
import os
from groq import Groq

def gen_groq(
    prompt: str,
    model: str = "openai/gpt-oss-20b",  # or "mixtral-8x7b-32768", etc.
    k: int = 3,
    temps = (0.6, 0.8, 1.0),
    max_tokens: int = 120,
    stream: bool = False,
) -> List[str]:
    """
    Generate k completions from Groq Chat API.
    Each with varying temperature. Returns a list of answer strings.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    outs = []
    for i in range(k):
        t = temps[i % len(temps)]
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(t),
            max_completion_tokens=max_tokens,
            top_p=0.95,
            stream=stream,
        )
        if stream:
            # Collect streamed chunks into one string
            text = ""
            for chunk in completion:
                text += chunk.choices[0].delta.content or ""
            outs.append(text.strip())
        else:
            outs.append(completion.choices[0].message.content.strip())
    return outs


# Example non-streaming implementation using environment variable for the API key.
# Keep secrets out of code; set GROQ_API_KEY in your environment (or CI secrets).
# def gen_groq(
#     prompt: str,
#     model: str = "openai/gpt-oss-20b",
#     k: int = 3,
#     temps = (0.6, 0.8, 1.0),
#     max_tokens: int = 120,
#     timeout: int = 60,
# ) -> list[str]:
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise RuntimeError("Set GROQ_API_KEY to use Groq provider.")
#     # ... perform requests with Authorization: Bearer <api_key> ...
#     return []
