
ANSWER_SYSTEM = (
    "You are a careful assistant in an ongoing chat.\n"
    "GROUNDING (facts):\n"
    "- Any fact about the documents MUST come from the numbered Sources. "
    "Do not use outside knowledge for facts.\n"
    "- If the question asks for a fact the Sources don't contain, reply exactly: "
    "INSUFFICIENT_CONTEXT\n"
    "- Cite the source numbers you used in square brackets, e.g. [1][3].\n"
    "CONVERSATION:\n"
    "- Use 'Conversation so far' to understand follow-ups and resolve "
    "references like 'it', 'that', 'the second one'.\n"
    "- You may answer questions about the conversation itself (e.g. what was "
    "asked earlier) from that history, no citation needed.\n"
    "- 'Earlier notes' are summaries of past chats for context only - never "
    "treat them as factual sources.\n"
    "- Be concise. Do not invent names, numbers, or dates."
)

def build_answer_prompt(
    question: str,
    chunks: list[dict],
    history_text: str = "",
    memories: list[str] | None = None,
) -> str:
    parts = []
    if history_text:
        parts.append(f"Conversation so far:\n{history_text}")
    if memories:
        notes = "\n".join(f"- {m}" for m in memories)
        parts.append(f"Earlier notes (context only):\n{notes}")

    if chunks:
        blocks = []
        for i, c in enumerate(chunks, start=1):
            src = f"{c['metadata']['source']} ({c['metadata']['locator']})"
            blocks.append(f"[{i}] source: {src}\n{c['text']}")
        parts.append("Sources:\n" + "\n\n".join(blocks))
    else:
        parts.append("Sources:\n(none retrieved)")

    parts.append(f"Question: {question}\n\nAnswer:")
    return "\n\n".join(parts)
CLARIFY_SYSTEM = (
    "You decide whether a user's question is answerable from the given context "
    "or whether it is too ambiguous and needs a clarifying question. "
    'Respond ONLY as JSON: {"clarify": true|false, "question": "..."}. '
    "Set clarify to true only if a short follow-up would materially help."
)

def build_clarify_prompt(question: str, chunks: list[dict]) -> str:
    snippets = "\n".join(f"- {c['text'][:160]}" for c in chunks) or "(no strong matches)"
    return (
        f"User question: {question}\n\n"
        f"Top retrieved snippets:\n{snippets}\n\n"
        "Is the question ambiguous or under-specified given these snippets?"
    )
CONTEXTUALIZE_SYSTEM = (
    "Given a chat history and a follow-up message, rewrite the follow-up as a "
    "standalone question understandable without the history. If it is already "
    "standalone, return it unchanged. Return ONLY the question, nothing else."
)
def build_contextualize_prompt(question: str, history_text: str) -> str:
    return f"Chat history:\n{history_text}\n\nFollow-up: {question}\n\nStandalone question:"

TABLE_SYSTEM = (
    "You answer questions about a single data table using ONLY the data given.\n"
    "- If the user names a specific field (the amount, the date, who placed it), "
    "return ONLY that value.\n"
    "- If the user asks about the order/record/row itself without naming a "
    "field, return the FULL row as 'Column: value' pairs for every column.\n"
    "- When rows are shown sorted, the order being asked about (most recent, "
    "oldest, etc.) is the FIRST row.\n"
    "- If the field name isn't an exact column, pick the closest column - e.g. "
    "'amount', 'price', 'value', 'cost' usually mean the monetary total column. "
    "Do NOT ask the user to clarify; choose the most relevant column.\n"
    "- Use the 'Facts' block ONLY for aggregates over all rows (total, count, "
    "average, min, max). For a specific row, read from the rows, never the Facts.\n"
    "- If the data doesn't contain the answer, reply exactly: INSUFFICIENT_CONTEXT\n"
    "- Never invent values. Be concise."
)
def build_table_prompt(
    question: str, source: str, table_csv: str, truncated: bool, facts: str = ""
) -> str:
    head = f"Table from {source}" + (" (first rows only)" if truncated else "")
    facts_block = f"Facts (precomputed - trust these):\n{facts}\n\n" if facts else ""
    return f"{facts_block}{head}:\n{table_csv}\nQuestion: {question}\nAnswer:"
