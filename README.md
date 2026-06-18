# RAG Chatbot

A small, self-hosted question-answering chatbot over your own documents
(PDF / Excel / CSV). It runs entirely on your machine: **Gemma 4** via Ollama
for generation, **sentence-transformers** for embeddings, and **ChromaDB** for
the vector store. Retrieval is **hybrid** (semantic + BM25), and the whole thing
is wrapped in guardrails so it fails *safely* — it refuses when the documents
don't support an answer instead of making something up. It also keeps
**conversation memory** (via LangChain) so follow-up questions and references
like "what about that?" work, and it remembers across visits.

The design goal here was correctness, grounding, and safe failure modes over
polish.

---

## Setup

**Prerequisites**

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

```bash
# 1. Pull the model (a few GB)
ollama pull gemma4:e4b

# 2. Set up the environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. create the env and tweak thresholds (optional)
# Example env

OLLAMA_HOST=http://localhost:11434
LLM_MODEL=gemma4:e4b
EMBED_MODEL=all-MiniLM-L6-v2

TOP_K=5
MIN_SCORE=0.25

CACHE_TTL_SECONDS=900

# 4. Run it
uvicorn main:app --reload
```

Open http://localhost:8000 — upload a few documents, pick an access level, then
ask questions. The first request is slow because the embedding model loads into
memory; after that it's quick.

> The first run also downloads the embedding model `all-MiniLM-L6-v2`

## Architecture overview

```
                 ┌──────────────┐
  Browser  ◀────▶│   FastAPI    │
 (static UI)     │   main.py    │
                 └──────┬───────┘
            ┌───────────┴────────────┐
            ▼                        ▼
     /ingest (routes)          /chat (routes)
            │                        │
   load → chunk → embed     cache → access-control
            │                        │
            ▼                  hybrid retrieval (RRF)
   ┌─────────────────┐         ┌──────┴───────┐
   │  ChromaDB (vec) │◀────────│ semantic     │
   │  BM25 (pickle)  │◀────────│ + keyword    │
   └─────────────────┘         └──────┬───────┘
                                grounding gate
                                ambiguity gate
                                      │
                                Gemma 4 (Ollama)
                                      │
                                output guardrails
                                (PII mask, leak scrub)
                                      │
                              answer + reference chunks
```


Used
**Folder layout**

```
main.py            FastAPI app, mounts routes + static UI
config.py          all tunables in one place
routes/            ingest, chat (+ chat/reset), health endpoints
utils/             the actual machinery (loaders, search, guardrails, memory...)
static/            vanilla HTML/CSS/JS frontend
data/              uploads + persisted Chroma store + BM25 pickle

```

**Request flow for `/chat`:** load short-term history → cache lookup (standalone
questions only) → resolve role to allowed access levels → rewrite follow-ups
into a standalone query → hybrid retrieval → recall long-term memory →
grounding gate → ambiguity gate → LLM answer (with history + memory + sources)
→ output guardrails → save the turn to memory → return answer with reference
chunks.

**Structured queries over spreadsheets:** similarity search can't answer
analytical questions like "the most recent order" or "total sales" — those need
sorting/aggregation, not look-alike chunks. So when a question is analytical
(`recent / latest / oldest / total / highest / count / how many ...`) and an
accessible spreadsheet exists, the request is routed to a structured path
(`utils/table_qa.py`): it loads the actual table, sorts it by a detected date
column (for "most recent/oldest"), precomputes the facts LLMs get wrong (row
count, sums, min/max/mean), and hands those plus the rows to the model. If the
table can't answer it, it falls back to normal RAG.

**Conversation memory (LangChain):**
- *Short-term* — a per-thread `InMemoryChatMessageHistory` (LangChain). The last
  few turns go into the prompt, and a follow-up like "and remote work?" is
  rewritten into a standalone question *before* retrieval so the search has real
  terms to match.
- *Long-term* — completed Q&A is embedded into a separate Chroma collection
  (`chat_memory`) and recalled by similarity on later questions. It's scoped per
  `user_id`, so one person's history never leaks into another's. Long-term
  recall is fed to the model as *context only* — never as a citable source — so
  it can't become a grounding/hallucination hole.
- `user_id` is stable per browser (localStorage); `session_id` is per chat and
  resets with the **New chat** button (clears short-term, keeps long-term).

---

## Key design choices & tradeoffs

- **Hybrid search with Reciprocal Rank Fusion.** Semantic search is great for
  paraphrases; BM25 is great for exact terms, IDs, and rare keywords. RRF fuses
  the two *rankings* (not raw scores), which sidesteps the headache of
  normalizing cosine similarity against BM25's unbounded scores. Tradeoff: RRF
  ignores score magnitude, so we keep the raw cosine score around separately for
  the grounding decision.
- **Grounding decision uses the raw signals, not the fused score.** The fused
  RRF value isn't a calibrated "how relevant is this" number, so we decide
  refuse-vs-answer on the underlying signals: pass if the best cosine similarity
  clears `MIN_SCORE`, *or* if BM25 actually matched query terms (a real keyword
  hit is evidence even when embedding similarity is lukewarm — common for short
  factual lookups). A query with neither signal is refused.
- **BM25 lives in a pickle, rebuilt on ingest.** `rank_bm25` has no incremental
  add, so we rebuild the index from the full corpus after each upload. For a
  local single-user corpus that's cheap and keeps the code simple.
- **Two-stage ambiguity check.** A cheap heuristic gate runs first (short query,
  pronoun-heavy, or flat retrieval scores); only then do we spend an LLM call to
  decide whether to ask a clarifying question. Saves a model round-trip on the
  common case.
- **`requests` instead of the Ollama SDK.** One fewer dependency; the HTTP API
  is trivial.
- **Embedder loaded once as a module singleton.** It's slow to initialize, so we
  never reload it per request.
- **Memory is context, sources are truth.** Long-term recall is injected as
  background notes the model is told not to cite, keeping the grounding contract
  intact while still giving conversational continuity. The grounding gate also
  relaxes (instead of hard-refusing on weak retrieval) *only* when there's
  conversation context, so meta questions like "what did I just ask?" work while
  unsupported factual questions are still refused.
- **Only `langchain-core` from LangChain.** Pulling `langchain-chroma` forced an
  incompatible `chromadb`/`tokenizers` combo, so long-term memory is backed by
  our existing Chroma stack and LangChain provides the chat-history primitives.
- **Cache is disabled once a thread has history.** A cached reply would be wrong
  for a context-dependent follow-up; standalone questions still cache.
- **Analytical table questions bypass vector search.** Embeddings can't rank
  "most recent" or compute a sum, so those route to a structured pandas path and
  the counts/totals are precomputed in code (the model only reads them off) —
  LLMs miscount rows otherwise. Sorting is done in plain Python on purpose: this
  environment's numpy/pandas build crashes on `DataFrame.sort_values` over a
  datetime column, and a list sort can't hit that bug.

---


## Guardrails implemented

1. **Grounded-only answering.** The system prompt forbids outside knowledge and
   requires the model to cite chunk numbers.
2. **Refuse when sources don't support an answer.** If retrieval is too weak
   (`MIN_SCORE`) or the model emits `INSUFFICIENT_CONTEXT`, we return an honest
   refusal instead of guessing.
3. **Ambiguity → clarifying question.** Under-specified questions get a follow-up
   rather than a confident-but-wrong answer.
4. **Access-restriction simulation.** Roles (`public` / `employee` / `manager`)
   map to clearance levels; the retriever only ever sees chunks at or below the
   caller's clearance, so confidential content never enters the prompt for an
   unauthorized role.
5. **Sensitive-information masking.** Emails, phone numbers, SSNs, card numbers,
   and API-key-shaped strings are redacted from the final answer.
6. **Private-info leak scrub.** Sentences carrying internal-only markers are
   dropped from the answer as a second layer behind access control.

---

## Evaluation approach

This is meant to be evaluated on *behavior*, not BLEU scores. Suggested checks:

- **Grounding / no-hallucination:** ask a question the documents *don't* answer
  → expect a refusal, not an invented answer.
- **Citations:** every answer should reference chunk numbers that actually
  appear in the returned `reference_chunks`.
- **Access control:** upload a `manager`-level doc, then query as `public`
  → the manager content must not appear in the answer or references.
- **PII masking:** include an email/phone in a source and ask for it
  → it should come back redacted.
- **Ambiguity:** ask a vague one-word question → expect a clarifying question.
- **Cache:** ask the same question twice → second response has `"cached": true`.
- **Short-term memory:** ask about vacation, then "and what about remote work?"
  → the follow-up is resolved and answered without repeating the topic.
- **Long-term memory:** after a few chats, click **New chat**, then ask
  something related → past Q&A is recalled as context. A different `user_id`
  must not recall your history.
- **Structured table queries:** upload an orders spreadsheet, ask "the most
  recent order" / "how many orders" / "total sales" → answers come from the
  sorted table and precomputed facts, not from random chunks.

A small labelled set (question, expected behavior) run against `/chat` is the
natural next step; the response shape (`type`, `reference_chunks`, `guardrails`)
is structured specifically to make automated assertions easy.

---

## Limitations

- Retrieval quality is bounded by a small embedding model and naive
  character-window chunking — tables and multi-column PDFs can chunk poorly.
- PII/private-info detection is heuristic; it will miss novel formats and can
  over-redact (e.g. long ID numbers that look like cards).
- The cache is exact-match on the normalized question, so "What is X?" and
  "Tell me about X" don't share a cache entry.
- BM25 rebuild is O(corpus) per ingest; fine for hundreds of docs, not millions.
- No streaming responses, no auth, no multi-tenant isolation.
- Short-term history lives in process memory, so it's lost on restart (long-term
  recall survives since it's persisted in Chroma).
- `user_id` comes from the browser and is trusted as-is — fine for a local demo,
  not a real isolation boundary.
- Structured table queries read only the first sheet, handle one table per
  question, and rely on keyword intent detection — complex filtered analytics
  ("orders from the East region over $500") aren't supported yet. "Most
  recent/oldest" returns the single top order (so a specific field like the
  amount is read off the right row deterministically); it doesn't return a list
  of the N most recent.

---

## Future improvements

- Semantic / embedding-based cache so paraphrases hit the same entry.
- A cross-encoder re-ranker over the fused candidates before answering.
- Incremental BM25 (or swap to a store that does hybrid natively).
- Proper PII detection (e.g. Presidio) and a trained confidentiality classifier.
- Streaming token output and per-source highlighting in the UI.
- Real authentication behind the simulated access levels.

## Model Selection

### Generation Model

This project uses **Gemma 4 (via Ollama)** as the primary LLM for question answering and reasoning.

### Why Gemma 4?

Gemma 4 was selected because it provides a strong balance between reasoning quality, instruction following, latency, and local deployment requirements.

Key reasons:

- **Open-source and self-hostable** — aligns with enterprise requirements where sensitive documents cannot be sent to external APIs.
- **Strong instruction-following capabilities** — important for grounded RAG workflows and citation-based responses.
- **Good reasoning performance for its size** — capable of answering questions using retrieved context while remaining lightweight enough for local execution.
- **Low operational complexity** — can be run entirely through Ollama without additional infrastructure.
- **Cost-efficient** — no external API costs and suitable for offline deployments.

### Embedding Model

The retrieval pipeline uses **all-MiniLM-L6-v2** from Sentence Transformers.

Reasons for selection:

- Fast embedding generation
- Small memory footprint
- Good semantic retrieval performance for business documents
- Well-suited for lightweight RAG systems

### Tradeoffs

While larger models may provide stronger reasoning capabilities, they require significantly more compute resources and increase response latency.

For this assessment, retrieval quality, grounding, and reliable failure behavior were prioritized over maximizing model size. Gemma 4 provided a practical balance between answer quality, speed, and ease of deployment.


## Demo Video

[Watch Demo Video](https://drive.google.com/file/d/1Es-FOYRkR-AdPUl5hAWf7Bf2XHlaQSIB/view?usp=sharing)
```
