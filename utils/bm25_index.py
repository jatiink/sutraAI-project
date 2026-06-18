from rank_bm25 import BM25Okapi
from config import BM25_PATH
import pickle
import re

_index = None
_records = []

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def rebuild(chunks: list[dict]) -> None:
    """Rebuild from the full corpus and persist."""
    global _index, _records
    _records = [{"id": c["id"], "text": c["text"], "metadata": c["metadata"]} for c in chunks]
    if _records:
        _index = BM25Okapi([tokenize(r["text"]) for r in _records])
    else:
        _index = None
    with open(BM25_PATH, "wb") as f:
        pickle.dump(_records, f)

def ensure_loaded() -> None:
    global _index, _records
    if _index is not None or not BM25_PATH.exists():
        return
    with open(BM25_PATH, "rb") as f:
        _records = pickle.load(f)
    if _records:
        _index = BM25Okapi([tokenize(r["text"]) for r in _records])

def search(query: str, top_k: int, allowed_levels: list[str]) -> list[dict]:
    ensure_loaded()
    if _index is None:
        return []
    scores = _index.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = []
    for i in ranked:
        if scores[i] <= 0:
            break
        rec = _records[i]
        if rec["metadata"].get("access_level") not in allowed_levels:
            continue
        hits.append({**rec, "score": float(scores[i])})
        if len(hits) >= top_k:
            break
    return hits
