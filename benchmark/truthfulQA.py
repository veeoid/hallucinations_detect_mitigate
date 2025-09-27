# d:/Hallucinations_GenAI/benchmark/truthfulQA.py
from datasets import load_dataset
import pandas as pd

# 1) Download & cache TruthfulQA (generation task)
ds = load_dataset("truthful_qa", "generation")["validation"]   # ~817 items

# 2) (Optional) reduce size while prototyping
N = 200
ds = ds.shuffle(seed=42).select(range(min(N, len(ds))))

# 3) Build rows with our own IDs
rows = []
for i, ex in enumerate(ds):
    q = (ex.get("question") or "").strip()
    ref = (ex.get("best_answer") or "").strip()   # loose reference for quick scoring
    rows.append({
        "id": f"val-{i:04d}",
        "question": q,
        "reference": ref
    })

# 4) Save to CSV your runner expects
out_path = "data/qa_mini.csv"
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"✅ Saved {len(rows)} rows to {out_path}")
