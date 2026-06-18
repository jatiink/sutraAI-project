import re

_PII_PATTERNS = [
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}\b")),
    ("api_key", re.compile(r"\b(?:sk|pk|api)[-_][A-Za-z0-9]{16,}\b")),
]
_PRIVATE_MARKERS = re.compile(
    r"\b(confidential|internal[ -]?only|do not distribute|proprietary|nda)\b",
    re.IGNORECASE,
)

def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, pattern in _PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[REDACTED_{label.upper()}]", text)
    return text, found

def scrub_private(text: str) -> tuple[str, bool]:
    if not _PRIVATE_MARKERS.search(text):
        return text, False
    kept = [s for s in re.split(r"(?<=[.!?])\s+", text) if not _PRIVATE_MARKERS.search(s)]
    cleaned = " ".join(kept).strip()
    if not cleaned:
        cleaned = "I can't share that - it looks like internal/confidential information."
    return cleaned, True

def apply(answer: str, scrub_internal: bool = True) -> dict:
    leaked = False
    if scrub_internal:
        answer, leaked = scrub_private(answer)
    answer, masked = mask_pii(answer)
    return {
        "answer": answer,
        "pii_masked": masked,
        "private_filtered": leaked,
    }
