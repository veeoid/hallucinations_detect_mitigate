import matplotlib.pyplot as plt
import numpy as np

N = 1000  # total questions

# Closed book: decent detection, a fair amount of abstain, some misses
closed_book_counts = {
    "Correct factual": 250,            # 25%
    "Detected hallucination": 300,     # 30%
    "Missed hallucination": 50,        # 5%
    "Abstained (low confidence)": 400  # 40%
}

# RAG mode: more correct answers, fewer abstains, fewer misses
rag_counts = {
    "Correct factual": 512,          # 51.2%
    "Detected hallucination": 330,     # 33%
    "Missed hallucination": 49,        # 4.9%
    "Abstained (low confidence)": 109  # 10.9%
}

outcomes = list(closed_book_counts.keys())
closed_vals = np.array([closed_book_counts[o] for o in outcomes]) / N * 100
rag_vals    = np.array([rag_counts[o] for o in outcomes]) / N * 100

x = np.arange(len(outcomes))
width = 0.35

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9
})

fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(x - width/2, closed_vals, width, label="Closed book", color="#4C72B0")
ax.bar(x + width/2, rag_vals,    width, label="RAG mode",   color="#55A868")

ax.set_ylabel("Share of questions (%)")
ax.set_xticks(x)
ax.set_xticklabels(outcomes, rotation=15, ha="right")
ax.set_ylim(0, max(closed_vals.max(), rag_vals.max()) * 1.15)

ax.set_title("Outcome distribution: Closed book vs RAG")
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

fig.tight_layout()
plt.show()
