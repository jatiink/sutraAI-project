from utils.prompts import ANSWER_SYSTEM, build_answer_prompt
from fastapi import APIRouter
from pydantic import BaseModel

from config import MIN_SCORE
from utils import (
    access_control,
    ambiguity,
    cache,
    guardrails,
    hybrid_search,
    llm,
    memory,
    table_qa,
)

router = APIRouter()

REFUSAL = (
    "I couldn't find anything in the uploaded documents that answers this. "
    "I'd rather not guess - try rephrasing or upload a source that covers it."
)


class ChatRequest(BaseModel):
    question: str
    role: str = "public"
    session_id: str = "default"   
    user_id: str = "anon"        


class ResetRequest(BaseModel):
    session_id: str = "default"


def _references(chunks: list[dict]) -> list[dict]:
    refs = []
    for i, c in enumerate(chunks, start=1):
        refs.append(
            {
                "n": i,
                "source": c["metadata"]["source"],
                "locator": c["metadata"]["locator"],
                "snippet": c["text"][:240] + ("..." if len(c["text"]) > 240 else ""),
                "scores": c["scores"],
                "rrf": round(c["rrf"], 4),
            }
        )
    return refs


def _best_semantic(chunks: list[dict]) -> float:
    sem = [c["scores"].get("semantic", 0.0) for c in chunks]
    return max(sem) if sem else 0.0


def _is_grounded(chunks: list[dict]) -> bool:
    if not chunks:
        return False
    if _best_semantic(chunks) >= MIN_SCORE:
        return True
    return any("keyword" in c["scores"] for c in chunks)


@router.post("/chat/reset")
def reset(req: ResetRequest):
    memory.reset_session(req.session_id)
    return {"status": "ok"}

@router.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    role = access_control.normalize_role(req.role)
    if not question:
        return {"type": "error", "answer": "Empty question."}
    history = memory.recent_pairs(req.session_id)
    history_text = memory.format_history(history)

    if not history_text:
        cached = cache.get(question, role)
        if cached:
            return {**cached, "cached": True}
    allowed = access_control.allowed_levels(role)
    search_query = memory.contextualize(question, history_text)

    if table_qa.is_analytical(search_query):
        table = table_qa.answer(search_query, allowed)
        if table:
            safe = guardrails.apply(table["answer"], scrub_internal=(role == "public"))
            response = {
                "type": "answer",
                "answer": safe["answer"],
                "reference_chunks": [
                    {
                        "n": 1,
                        "source": table["source"],
                        "locator": f"structured query over {table['rows']} rows",
                        "snippet": "(answered from the full table, not a text chunk)",
                        "scores": {},
                        "rrf": 0,
                    }
                ],
                "guardrails": {
                    "pii_masked": safe["pii_masked"],
                    "private_filtered": safe["private_filtered"],
                },
                "cached": False,
            }
            memory.remember_turn(req.session_id, req.user_id, question, safe["answer"])
            return response
    chunks = hybrid_search.search(search_query, allowed)
    grounded = _is_grounded(chunks)
    memories = memory.recall(search_query, req.user_id)
    if not grounded and not history_text and not memories:
        return {"type": "refusal", "answer": REFUSAL, "reference_chunks": [], "cached": False}

    if grounded:
        clarifying = ambiguity.maybe_clarify(search_query, chunks)
        if clarifying:
            return {
                "type": "clarification",
                "answer": clarifying,
                "reference_chunks": _references(chunks),
                "cached": False,
            }
    source_chunks = chunks if grounded else []
    try:
        raw = llm.generate(
            build_answer_prompt(question, source_chunks, history_text, memories),
            system=ANSWER_SYSTEM,
        )
    except llm.LLMError as e:
        return {"type": "error", "answer": str(e)}

    refs = _references(source_chunks)
    if "INSUFFICIENT_CONTEXT" in raw:
        memory.remember_turn(req.session_id, req.user_id, question, REFUSAL, persist=False)
        return {"type": "refusal", "answer": REFUSAL, "reference_chunks": refs, "cached": False}
    safe = guardrails.apply(raw, scrub_internal=(role == "public"))
    response = {
        "type": "answer",
        "answer": safe["answer"],
        "reference_chunks": refs,
        "guardrails": {
            "pii_masked": safe["pii_masked"],
            "private_filtered": safe["private_filtered"],
        },
        "cached": False,
    }
    memory.remember_turn(req.session_id, req.user_id, question, safe["answer"])
    if not history_text:
        cache.set(question, role, response)
    return response
