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

def gen_groq(prompt: str,
             model: str = "llama-3.1-8b-instant",
             k: int = 3,
             temps = 1,
             max_tokens: int = 120) -> list[str]:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY env var not set")
    client = Groq(api_key=api_key)

    outs = []
    for i in range(k):
        t = temps[i % len(temps)]
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt+ "\n Only respond with the final answer. \n IF YOU ARE UNSURE, RESPOND WITH 'UNKNOWN'."}],
            temperature=t,
            max_tokens=max_tokens,
            top_p=1,
            stream=False,
        )
        outs.append(resp.choices[0].message.content.strip())
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
