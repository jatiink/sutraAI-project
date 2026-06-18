import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL
_model = None
_client = None
_collection = None

def _embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def _coll():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection

def get_collection(name: str):
    _coll()
    return _client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

def embed(texts: list[str]) -> list[list[float]]:
    vecs = _embedder().encode(texts, normalize_embeddings=True)
    return vecs.tolist()

def add_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return
    _coll().add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[
            {
                "source": c["source"],
                "locator": c["locator"],
                "access_level": c["access_level"],
            }
            for c in chunks
        ],
    )

def delete_by_source(source: str) -> None:
    _coll().delete(where={"source": source})

def semantic_search(query: str, top_k: int, allowed_levels: list[str]) -> list[dict]:
    if count() == 0:
        return []
    res = _coll().query(
        query_embeddings=embed([query]),
        n_results=top_k,
        where={"access_level": {"$in": allowed_levels}},
    )
    hits = []
    ids = res["ids"][0]
    for i, _id in enumerate(ids):
        dist = res["distances"][0][i]
        hits.append(
            {
                "id": _id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": 1.0 - dist,  # cosine distance -> similarity
            }
        )
    return hits

def all_chunks() -> list[dict]:
    data = _coll().get()
    out = []
    for i, _id in enumerate(data["ids"]):
        out.append(
            {
                "id": _id,
                "text": data["documents"][i],
                "metadata": data["metadatas"][i],
            }
        )
    return out





def count() -> int:
    return _coll().count()
