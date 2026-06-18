from utils import llm
from utils.prompts import CLARIFY_SYSTEM, build_clarify_prompt

_VAGUE_WORDS = {"it", "this", "that", "they", "them", "thing", "stuff"}
def looks_ambiguous(question: str, chunks: list[dict]) -> bool:
    q = question.strip()
    words = q.lower().split()
    if len(words) <= 2:
        return True
    if words and words[0] in _VAGUE_WORDS:
        return True
    sem = [c["scores"].get("semantic", 0.0) for c in chunks if "semantic" in c["scores"]]
    if len(sem) >= 2 and (max(sem) - min(sem)) < 0.05:
        return True
    return False

def maybe_clarify(question: str, chunks: list[dict]) -> str | None:
    if not looks_ambiguous(question, chunks):
        return None
    result = llm.generate_json(
        build_clarify_prompt(question, chunks), system=CLARIFY_SYSTEM
    )
    if result.get("clarify") and result.get("question"):
        return result["question"].strip()
    return None
