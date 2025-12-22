from groq import Groq
from dotenv import load_dotenv
import os

client = Groq()
load_dotenv()  # allow .env GROQ_API_KEY
api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
      {
        "role": "user",
        "content": "who is the president of the united states?"
      },
      {
        "role": "user",
        "content": ""
      }
    ],
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
