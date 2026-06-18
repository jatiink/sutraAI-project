"""Quality metrics computed from a /chat response.

These are deliberately cheap heuristics - no second LLM as a judge - so the
eval runs offline in seconds. They're approximate, but good enough to catch
regressions in grounding, citations, and obvious hallucinations.
"""
import re

_NUM = re.compile(r"\d+(?:\.\d+)?")
_CITE = re.compile(r"\[\d+\]")


def _is_table_answer(resp: dict) -> bool:
    return any("structured query" in r.get("locator", "") for r in resp.get("reference_chunks", []))


def best_semantic(resp: dict) -> float:
    scores = [
        r["scores"].get("semantic", 0.0)
        for r in resp.get("reference_chunks", [])
        if r.get("scores")
    ]
    return max(scores) if scores else 0.0


def groundedness(resp: dict) -> float:
    """Fraction of numbers in the answer that actually appear in the sources.

    Numbers are where hallucinations hurt most (wrong amount, wrong count), and
    they're easy to verify against the retrieved chunk text. 1.0 = every number
    is supported (or the answer has no numbers). Table answers are grounded by
    construction (read straight from the sheet), so they score 1.0.
    """
    answer = resp.get("answer", "")
    nums = _NUM.findall(answer.replace(",", ""))
    if not nums or _is_table_answer(resp):
        return 1.0
    haystack = " ".join(r.get("snippet", "") for r in resp.get("reference_chunks", [])).replace(",", "")
    supported = sum(1 for n in nums if n in haystack)
    return round(supported / len(nums), 2)


def has_citation(resp: dict) -> bool:
    # table answers are deterministically sourced, so we count them as cited
    return _is_table_answer(resp) or bool(_CITE.search(resp.get("answer", "")))


def confidence(resp: dict) -> dict | None:
    """0-1 confidence for an answer, blended from retrieval + citation + grounding.

    Returns None for refusals/clarifications (nothing was asserted to score).
    """
    if resp.get("type") != "answer":
        return None
    retrieval = 1.0 if _is_table_answer(resp) else best_semantic(resp)
    cite = 1.0 if has_citation(resp) else 0.0
    ground = groundedness(resp)
    score = round(0.5 * retrieval + 0.2 * cite + 0.3 * ground, 2)
    label = "high" if score >= 0.66 else "medium" if score >= 0.4 else "low"
    return {"score": score, "label": label,
            "retrieval": round(retrieval, 2), "citation": cite, "groundedness": ground}
