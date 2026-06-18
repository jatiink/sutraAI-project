from config import CACHE_MAX_ENTRIES, CACHE_TTL_SECONDS
import re
import time
from collections import OrderedDict

_store: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
def _key(question: str, role: str) -> str:
    norm = re.sub(r"\s+", " ", question.strip().lower())
    return f"{role}::{norm}"

def get(question: str, role: str) -> dict | None:
    key = _key(question, role)
    item = _store.get(key)
    if item is None:
        return None
    ts, value = item
    if time.time() - ts > CACHE_TTL_SECONDS:
        _store.pop(key, None)
        return None
    _store.move_to_end(key) 
    return value

def set(question: str, role: str, value: dict) -> None:
    key = _key(question, role)
    _store[key] = (time.time(), value)
    _store.move_to_end(key)
    while len(_store) > CACHE_MAX_ENTRIES:
        _store.popitem(last=False)

def clear() -> None:
    _store.clear()
