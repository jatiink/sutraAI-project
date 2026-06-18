from pathlib import Path
import pandas as pd
from pypdf import PdfReader

SUPPORTED = {".pdf", ".xlsx", ".xls", ".csv"}

class UnsupportedFile(Exception):
    pass

def load(path: str | Path) -> list[dict]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in (".xlsx", ".xls"):
        return _load_excel(path)
    if ext == ".csv":
        return _load_csv(path)
    raise UnsupportedFile(f"can't handle {ext} files (yet)")

def _load_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append({"text": text, "locator": f"page {i}"})
    return sections
ROWS_PER_SECTION = 1

def _load_excel(path: Path) -> list[dict]:
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    sections = []
    for name, df in sheets.items():
        sections.extend(_frame_to_sections(df, label=f"sheet '{name}'"))
    return sections

def _load_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str)
    return _frame_to_sections(df, label="csv")

def _frame_to_sections(df: pd.DataFrame, label: str) -> list[dict]:
    df = df.fillna("")
    cols = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        parts = [f"{c}: {row[c]}" for c in cols if str(row[c]).strip()]
        if parts:
            rows.append(" | ".join(parts))

    sections = []
    for start in range(0, len(rows), ROWS_PER_SECTION):
        block = rows[start : start + ROWS_PER_SECTION]
        sections.append({"text": "\n".join(block), "locator": _locator(label, start, len(block))})
    return sections

def _locator(label: str, start: int, size: int) -> str:
    if size == 1:
        return f"{label} row {start + 1}"
    return f"{label} rows {start + 1}-{start + size}"
