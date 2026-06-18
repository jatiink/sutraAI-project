"""Offline eval harness. Run from the project root:

    ./.venv/bin/python -m eval.evaluate

Spins up an *isolated* index in a temp dir (so it never touches your real
data/), ingests the fixtures, runs every labeled case through the real /chat
pipeline, and prints per-case results + an aggregate summary. Needs Ollama up.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- isolate the index BEFORE the pipeline touches it -----------------------
# The util modules bind these names at import time, so we patch the modules
# (not just config) to point at a throwaway dir.
_TMP = Path(tempfile.mkdtemp(prefix="rag_eval_"))
import utils.vector_store as vstore
import utils.bm25_index as bm25
import utils.table_qa as table_qa

(_TMP / "chroma").mkdir(parents=True, exist_ok=True)
vstore.CHROMA_DIR = _TMP / "chroma"
bm25.BM25_PATH = _TMP / "chroma" / "bm25.pkl"
table_qa.REGISTRY = _TMP / "tables.json"

from eval.dataset import CASES          # noqa: E402
from eval.metrics import confidence, groundedness, has_citation  # noqa: E402
from routes.chat import ChatRequest, chat  # noqa: E402
from utils import chunker, document_loader  # noqa: E402

FIXTURES = [
    ("handbook.csv", "public"),
    ("salaries.csv", "manager"),
    ("orders.csv", "public"),
]


def setup_index():
    for name, level in FIXTURES:
        path = ROOT / "eval" / "fixtures" / name
        sections = document_loader.load(path)
        chunks = chunker.chunk_sections(sections, source=name, access_level=level)
        vstore.add_chunks(chunks)
        table_qa.register(name, str(path), level)
    bm25.rebuild(vstore.all_chunks())


def run_case(case: dict) -> dict:
    resp = chat(ChatRequest(
        question=case["q"], role=case.get("role", "public"),
        session_id=case["id"], user_id=case["id"],
    ))
    sources = [r["source"] for r in resp.get("reference_chunks", [])]
    answer = resp.get("answer", "")
    checks = {}

    checks["behavior"] = resp.get("type") == case["expect_type"]

    if case.get("keywords"):
        hit = sum(1 for k in case["keywords"] if k.lower() in answer.lower())
        checks["keyword_recall"] = round(hit / len(case["keywords"]), 2)

    if case.get("source") and resp.get("type") == "answer":
        checks["retrieval_hit"] = case["source"] in sources

    if case.get("forbid_source"):
        checks["no_leak"] = case["forbid_source"] not in sources

    if case.get("pii"):
        masked = resp.get("guardrails", {}).get("pii_masked", [])
        checks["pii_masked"] = all(p in masked for p in case["pii"])

    if case.get("citation") and resp.get("type") == "answer":
        checks["citation"] = has_citation(resp)

    return {
        "id": case["id"],
        "type": resp.get("type"),
        "answer": answer[:70] + ("..." if len(answer) > 70 else ""),
        "checks": checks,
        "groundedness": groundedness(resp) if resp.get("type") == "answer" else None,
        "confidence": confidence(resp),
    }


def main():
    setup_index()
    print(f"Running {len(CASES)} cases...\n")

    results = [run_case(c) for c in CASES]

    passed = total = 0
    for r in results:
        flags = []
        for name, val in r["checks"].items():
            ok = val is True or (isinstance(val, (int, float)) and val >= 1.0)
            total += 1
            passed += ok
            flags.append(f"{name}={'ok' if ok else val}")
        conf = r["confidence"]
        conf_str = f"conf={conf['label']}({conf['score']})" if conf else "conf=n/a"
        print(f"[{r['id']:<7}] {r['type']:<13} {conf_str:<18} {'  '.join(flags)}")
        print(f"          -> {r['answer']}")

    g = [r["groundedness"] for r in results if r["groundedness"] is not None]
    print("\n--- summary ---")
    print(f"checks passed:     {passed}/{total} ({100 * passed // total}%)")
    print(f"behavior correct:  {sum(r['checks'].get('behavior', False) for r in results)}/{len(results)}")
    print(f"avg groundedness:  {round(sum(g) / len(g), 2) if g else 'n/a'}")


if __name__ == "__main__":
    main()
