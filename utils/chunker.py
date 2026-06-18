from config import CHUNK_OVERLAP, CHUNK_SIZE
import uuid

def chunk_sections(sections: list[dict], source: str, access_level: str) -> list[dict]:
    chunks = []
    for section in sections:
        for piece in _split(section["text"]):
            chunks.append(
                {
                    "id": uuid.uuid4().hex,
                    "text": piece,
                    "source": source,
                    "locator": section["locator"],
                    "access_level": access_level,
                }
            )
    return chunks

def _split(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []
    out = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    while start < len(text):
        window = text[start : start + CHUNK_SIZE]
        if start + CHUNK_SIZE < len(text):
            cut = window.rfind(" ")
            if cut > CHUNK_SIZE * 0.6:
                window = window[:cut]
        window = window.strip()
        if window:
            out.append(window)
        start += step
    return out
