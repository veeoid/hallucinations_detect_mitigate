# Hallucinations Detect & Mitigate

Small research/demo project that generates candidate answers using an LLM (the provided `generation.py` calls an Ollama endpoint) and scores semantic diversity / "semantic entropy" across multiple responses.

## What this repo contains

- `src/` — main scripts:
  - `runner_closed_book.py` — example runner that reads `data/qa_mini.csv`, generates k samples for the first question, and computes semantic entropy.
  - `generation.py` — small client that posts to your Ollama HTTP API (update endpoint in the file if needed).
  - `semantic_entropy.py` — computes embeddings and clusters to estimate normalized entropy. The module lazily loads `sentence-transformers` at runtime.
- `data/qa_mini.csv` — small sample dataset (tracked in repo).
- `benchmark/` — utility scripts (e.g., `truthfulQA.py`).
- `requirements.txt` — pinned runtime dependencies (created from the active venv).

## Prerequisites

- Windows (tested with PowerShell).
- Python 3.12 installed and on your PATH.
- Enough disk and network bandwidth: `requirements.txt` includes large packages (e.g., `torch`, `pyarrow`).
- An Ollama HTTP endpoint if you want to run `runner_closed_book.py` end-to-end. If you don't have Ollama running, `generation.py` will fail when it tries to POST to the local server.

## Quickstart (PowerShell)

Open PowerShell in the repo root (`D:\Hallucinations_GenAI`) and run:

```powershell
# create a fresh venv (optional if you already have .venv-hallu-detect)
python -m venv .venv-hallu-detect

# activate the venv (PowerShell)
& .\.venv-hallu-detect\Scripts\Activate.ps1

# upgrade pip and install pinned dependencies (may download hundreds of MBs)
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# run the runner (will call your Ollama endpoint)
python .\src\runner_closed_book.py
```

If you prefer to run the file using the venv python directly without activating:

```powershell
& .\.venv-hallu-detect\Scripts\python.exe .\src\runner_closed_book.py
```

## Notes

- requirements.txt pins the exact versions present in the venv this README was generated from. Installing will fetch large artifacts (torch, pyarrow, etc.). If you want a smaller/portable setup, create a minimal list of direct dependencies (e.g., `sentence-transformers`, `datasets`, `scikit-learn`, `pandas`, `requests`) and install only those.

- `semantic_entropy.py` is written to lazily import `SentenceTransformer`. If you see an ImportError mentioning `SentenceTransformer`, run:

```powershell
python -m pip install --force-reinstall sentence-transformers
```

- The project uses `intfloat/e5-small-v2` by default in `semantic_entropy.py`. You can change the model name inside that file if you want a different embedding model (smaller, larger, or local CPU-only model).

- `generation.py` posts to the Ollama API. If you don't run Ollama locally, you can replace the generation call with a stub that returns a list of canned responses for testing.


