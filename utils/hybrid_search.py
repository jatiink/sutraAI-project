from utils import bm25_index, vector_store
from config import RRF_K, TOP_K

def search(query: str, allowed_levels: list[str], top_k: int = TOP_K) -> list[dict]:
    # pull a few extra from each side so fusion has something to work with
    pool = max(top_k * 2, 10)
    semantic = vector_store.semantic_search(query, pool, allowed_levels)
    keyword = bm25_index.search(query, pool, allowed_levels)

    fused: dict[str, dict] = {}
    fuse(fused, semantic, "semantic")
    fuse(fused, keyword, "keyword")

    results = sorted(fused.values(), key=lambda c: c["rrf"], reverse=True)
    return results[:top_k]

def fuse(acc: dict, hits: list[dict], label: str) -> None:
    for rank, hit in enumerate(hits):
        entry = acc.get(hit["id"])
        if entry is None:
            entry = {
                "id": hit["id"],
                "text": hit["text"],
                "metadata": hit["metadata"],
                "rrf": 0.0,
                "scores": {},
            }
            acc[hit["id"]] = entry
        entry["rrf"] += 1.0 / (RRF_K + rank + 1)
        entry["scores"][label] = round(hit["score"], 4)
