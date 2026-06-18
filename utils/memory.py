
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
import uuid
import time

from config import (
    LONG_TERM_MIN_SCORE,
    LONG_TERM_TOP_K,
    MEMORY_COLLECTION,
    SHORT_TERM_TURNS,
)
from utils import llm, vector_store
from utils.prompts import CONTEXTUALIZE_SYSTEM, build_contextualize_prompt

_sessions: dict[str, InMemoryChatMessageHistory] = {}

def history_for(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _sessions:
        _sessions[session_id] = InMemoryChatMessageHistory()
    return _sessions[session_id]

def recent_pairs(session_id: str, limit: int = SHORT_TERM_TURNS) -> list[tuple[str, str]]:
    msgs = history_for(session_id).messages
    pairs = []
    pending = None
    for m in msgs:
        if isinstance(m, HumanMessage):
            pending = m.content
        elif isinstance(m, AIMessage) and pending is not None:
            pairs.append((pending, m.content))
            pending = None
    return pairs[-limit:]

def format_history(pairs: list[tuple[str, str]]) -> str:
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in pairs)

def remember_turn(session_id: str, user_id: str, question: str, answer: str, persist: bool = True) -> None:
    h = history_for(session_id)
    h.add_user_message(question)
    h.add_ai_message(answer)
    if persist:
        save_long_term(user_id, session_id, question, answer)

def reset_session(session_id: str) -> None:
    _sessions.pop(session_id, None)

def save_long_term(user_id: str, session_id: str, question: str, answer: str) -> None:
    text = f"Q: {question}\nA: {answer}"
    vector_store.get_collection(MEMORY_COLLECTION).add(
        ids=[uuid.uuid4().hex],
        documents=[text],
        embeddings=vector_store.embed([text]),
        metadatas=[{"user_id": user_id, "session_id": session_id, "ts": time.time()}],
    )

def recall(query: str, user_id: str, top_k: int = LONG_TERM_TOP_K) -> list[str]:
    coll = vector_store.get_collection(MEMORY_COLLECTION)
    if coll.count() == 0:
        return []
    res = coll.query(
        query_embeddings=vector_store.embed([query]),
        n_results=top_k,
        where={"user_id": user_id},
    )
    out = []
    for i, doc in enumerate(res["documents"][0]):
        if 1.0 - res["distances"][0][i] >= LONG_TERM_MIN_SCORE:
            out.append(doc)
    return out

def contextualize(question: str, history_text: str) -> str:
    if not history_text:
        return question
    rewritten = llm.generate(
        build_contextualize_prompt(question, history_text),
        system=CONTEXTUALIZE_SYSTEM,
        temperature=0.0,
    )
    if not rewritten or len(rewritten) > 300:
        return question
    return rewritten.strip()
