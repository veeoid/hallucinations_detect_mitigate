# src/retrieval_live.py
from __future__ import annotations
import os, json, hashlib
from typing import List, Tuple, Optional
import requests

# NEW: Import lightweight search API
from ddgs import DDGS

USER_AGENT = "HallucinationPipeline/1.0 (academic-use)"
CACHE_DIR = os.path.join("data", "corpus_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _slug(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def _cache_read(key: str) -> Optional[dict]:
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _cache_write(key: str, data: dict) -> None:
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _http_get(url: str, params: dict = None, timeout: float = 1.0) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, params=params or {}, timeout=timeout, headers={"User-Agent": USER_AGENT})
        if resp.status_code == 200:
            return resp
    except Exception:
        return None
    return None

# ------------------------------
# Wikipedia REST summary
# ------------------------------

def wiki_summary(query: str, timeout: float = 1.0) -> Optional[Tuple[str, str]]:
    """
    Wikipedia REST summary API (with normalization and title search).
    It first normalizes the query to a likely title, and if that fails,
    it uses the Wikipedia search API to find the closest page.
    Returns (text, source_url) or None.
    """
    def _normalize_question_to_title(q: str) -> str:
        s = q.strip().replace("?", "")
        s_low = s.lower()
        # remove question words
        for prefix in (
            "who is", "who was", "what is", "what was", "where is", "where was",
            "in what country was", "in which country was", "when was", "when did",
            "name the", "give the", "tell me", "please tell"
        ):
            if s_low.startswith(prefix + " "):
                s = s[len(prefix):].strip()
                break
        # drop trailing filler words
        for tail in ("born", "located", "founded", "made", "created"):
            if s.lower().endswith(tail):
                s = s[: -len(tail)].strip(",. ").strip()
        # capitalize first letter for Wikipedia
        return s.strip().capitalize() or q.strip()

    title = _normalize_question_to_title(query)
    key = f"wiki_{_slug(title)}"

    # Cache check
    cached = _cache_read(key)
    if cached:
        txt, url = cached.get("text"), cached.get("url")
        if txt:
            return txt, url

    # Direct summary attempt
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    print(f"[TRACE] Wikipedia API URL (normalized) for query \"{query}\": {url}")
    resp = _http_get(url, timeout=timeout)
    if resp and resp.status_code == 200:
        data = resp.json()
        extract = data.get("extract") or ""
        page_url = (data.get("content_urls") or {}).get("desktop", {}).get("page") \
                   or data.get("canonicalurl") or data.get("uri")
        if extract:
            _cache_write(key, {"text": extract, "url": page_url})
            return extract, page_url

    # Fallback: use Wikipedia search to find best page title
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": title,
        "srlimit": 1,
        "format": "json",
        "utf8": 1,
    }
    resp = _http_get(search_url, params=params, timeout=timeout)
    if not resp:
        print(f"[TRACE] Wikipedia search failed for query: {query}")
        return None
    hits = (((resp.json() or {}).get("query") or {}).get("search") or [])
    if hits:
        best_title = hits[0].get("title")
        if best_title:
            print(f"[TRACE] Wikipedia search resolved \"{query}\" → \"{best_title}\"")
            # Fetch summary by best title
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{best_title.replace(' ', '_')}"
            resp2 = _http_get(summary_url, timeout=timeout)
            if resp2 and resp2.status_code == 200:
                data2 = resp2.json()
                extract = data2.get("extract") or ""
                page_url = (data2.get("content_urls") or {}).get("desktop", {}).get("page") \
                           or data2.get("canonicalurl") or data2.get("uri")
                if extract:
                    _cache_write(key, {"text": extract, "url": page_url})
                    return extract, page_url

    print(f"[TRACE] Wikipedia: No summary found for query \"{query}\" (title tried: \"{title}\")")
    return None


# ------------------------------
# DuckDuckGo fallback using duckduckgo-search
# ------------------------------

def ddg_text_search(query: str, timeout: float = 2.0, max_results: int = 2) -> Optional[List[Tuple[str, str]]]:
    """
    Uses duckduckgo_search library to fetch short text snippets.
    Returns list of (url, snippet).
    """
    key = f"ddg_text_{_slug(query)}"
    cached = _cache_read(key)
    if cached and cached.get("results"):
        return cached["results"]

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                if r and r.get("body"):
                    results.append((r.get("href") or "DuckDuckGo", r["body"]))
        if results:
            _cache_write(key, {"results": results})
            return results
    except Exception as e:
        print(f"[TRACE] DDG text search error: {e}")
        return None
    return None

# ------------------------------
# Unified entry point
# ------------------------------

def retrieve_live_context(query: str, top_k: int = 2, timeout: float = 1.0) -> List[Tuple[str, str]]:
    """
    Try Wikipedia first, then DuckDuckGo text search. Returns list of (source_id, snippet).
    """
    contexts: List[Tuple[str, str]] = []
    seen = set()

    # Try Wikipedia
    w = wiki_summary(query, timeout=timeout)
    print(f"[TRACE] Wikipedia retrieval result for query \"{query}\": {w is not None}\n")
    if w and w[0]:
        src_id = w[1] or "Wikipedia"
        if src_id not in seen:
            contexts.append((src_id, w[0][:1200]))
            seen.add(src_id)
    print(f'We are calling the wiki_summary function and the result is {w}\n')
    print(f"[TRACE] Retrieved {len(contexts)} contexts from Wikipedia for query: \"{query}\"")

    print(f'Conexts after wiki retrieval:\n')
    for ctx in contexts:
        print(f'Source: {ctx[0]}\nSnippet: {ctx[1]}\n')

    # Fallback to DDG text search if Wikipedia failed or incomplete
    if len(contexts) < top_k:
        print(f'[TRACE] Since we have only {len(contexts)} contexts, we will call DDG text search for query: "{query}"\n')
        ddg_results = ddg_text_search(query, timeout=timeout, max_results=top_k)
        if ddg_results:
            for url, snippet in ddg_results:
                if url not in seen:
                    contexts.append((url, snippet[:1200]))
                    seen.add(url)
        print(f"[TRACE] Retrieved {len(contexts)} total contexts after DDG for query: \"{query}\"")
        for ctx in contexts:
            print(f'DDG Source: {ctx[0]}\nSnippet: {ctx[1]}\n')

    print(f"[TRACE] Final retrieved {len(contexts)} total contexts for query: \"{query}\"")
    return contexts[:top_k]
