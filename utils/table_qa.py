from utils.prompts import TABLE_SYSTEM, build_table_prompt
from utils import llm, vector_store
import csv
from config import DATA_DIR
import pandas as pd
import io
import json
import re

REGISTRY = DATA_DIR / "tables.json"
MAX_ROWS = 60 
_INTENT = re.compile(
    r"\b(recent|latest|newest|last|oldest|earliest|first|highest|largest|most|"
    r"biggest|lowest|smallest|least|top|bottom|total|sum|average|avg|mean|"
    r"count|how many|max|maximum|min|minimum)\b",
    re.I,
)
_RECENT = re.compile(r"\b(recent|latest|newest|last)\b", re.I)
_OLDEST = re.compile(r"\b(oldest|earliest|first)\b", re.I)

def is_analytical(query: str) -> bool:
    return bool(_INTENT.search(query))

def register(source: str, path, access_level: str) -> None:
    reg = _registry()
    reg[source] = {"path": str(path), "access_level": access_level}
    REGISTRY.write_text(json.dumps(reg, indent=2))

def _registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {}

def _read(path: str) -> pd.DataFrame:
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path, dtype=str)
    return pd.read_excel(path, dtype=str)  # first sheet is enough for now

def answer(query: str, allowed_levels: list[str]) -> dict | None:
    picked = _pick_table(query, allowed_levels)
    if not picked:
        return None
    source, meta = picked
    try:
        df = _read(meta["path"]).fillna("")
    except Exception:
        return None  
    cols = list(df.columns)
    records = df.to_dict("records")
    date_col = _first_date_col(df)
    is_temporal = bool(date_col and (_RECENT.search(query) or _OLDEST.search(query)))
    if is_temporal:
        parsed = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        keys = [k if pd.notna(k) else pd.Timestamp.min for k in parsed]
        target = min(
            zip(keys, records),
            key=lambda t: t[0],
            default=(None, None),
        )[1] if not _RECENT.search(query) else max(
            zip(keys, records), key=lambda t: t[0], default=(None, None)
        )[1]
        if target is None:
            return None
        field = _match_field(query, cols)
        ans = str(target[field]) if field else " | ".join(f"{c}: {target[c]}" for c in cols)
        return {"answer": ans, "source": source, "rows": 1}
    truncated = len(records) > MAX_ROWS
    table_csv = _to_csv(cols, records[:MAX_ROWS])
    raw = llm.generate(
        build_table_prompt(query, source, table_csv, truncated, _facts(df)),
        system=TABLE_SYSTEM,
        temperature=0.0,
    )
    if "INSUFFICIENT_CONTEXT" in raw:
        return None
    return {"answer": raw.strip(), "source": source, "rows": min(len(records), MAX_ROWS)}

def _pick_table(query: str, allowed_levels: list[str]):
    cands = [(s, m) for s, m in _registry().items() if m["access_level"] in allowed_levels]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    schemas = []
    for source, meta in cands:
        try:
            cols = ", ".join(_read(meta["path"]).columns)
        except Exception:
            cols = ""
        schemas.append(f"{source}: {cols}")
    qv = vector_store.embed([query])[0]
    svs = vector_store.embed(schemas)
    sims = [sum(a * b for a, b in zip(qv, sv)) for sv in svs]
    return cands[max(range(len(cands)), key=lambda i: sims[i])]

_FIELD_HINTS = {
    "money": (
        ["amount", "price", "cost", "value", "total", "revenue", "sales", "paid", "spent"],
        ["total", "amount", "revenue", "sales", "price", "value", "cost"],
    ),
    "person": (
        ["who", "rep", "salesperson", "seller", "person", "placed", "customer", "client"],
        ["rep", "name", "customer", "client", "person", "sales", "seller"],
    ),
    "date": (["date", "when", "day", "time"], ["date", "day", "time"]),
    "place": (
        ["region", "where", "area", "location", "country", "state", "city"],
        ["region", "area", "location", "country", "state", "city"],
    ),
    "quantity": (["units", "quantity", "qty"], ["units", "qty", "quantity", "count"]),
    "item": (["item", "product", "goods", "sku"], ["item", "product", "sku", "goods"]),
}

def _match_field(query: str, columns: list[str]) -> str | None:
    q = query.lower()
    by_lower = {c.lower(): c for c in columns}
    direct = [c for cl, c in by_lower.items() if cl in q]
    if len(direct) == 1:
        return direct[0]
    for keywords, patterns in _FIELD_HINTS.values():
        if any(k in q for k in keywords):
            for pat in patterns:
                for cl, c in by_lower.items():
                    if pat in cl:
                        return c
    return direct[0] if direct else None

def _first_date_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.7:
            return c
    return None

def _facts(df: pd.DataFrame) -> str:
    lines = [f"Total rows: {len(df)}"]
    for c in df.columns:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() >= 0.7:
            lines.append(
                f"{c}: sum={num.sum():.2f}, min={num.min():g}, "
                f"max={num.max():g}, mean={num.mean():.2f}"
            )
    return "\n".join(lines)

def _to_csv(columns: list[str], records: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue()
