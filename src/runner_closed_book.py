import pandas as pd
from generation import gen_ollama
from semantic_entropy import semantic_entropy

CSV_PATH = "data/qa_mini.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    q = df.iloc[0]["question"]
    print("Question:", q, "\n")

    samples = gen_ollama(q, k=3, max_tokens=120)
    for i, s in enumerate(samples, 1):
        print(f"Sample {i}: {s}\n")

    s_conf, clusters = semantic_entropy(samples)
    print("Semantic Entropy Score:", round(s_conf, 3))
    print("Clusters:", clusters)

if __name__ == "__main__":
    main()
