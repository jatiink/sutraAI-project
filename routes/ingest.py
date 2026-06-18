from utils import access_control, bm25_index, table_qa, vector_store
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from utils.chunker import chunk_sections
from config import UPLOAD_DIR
from utils.document_loader import SUPPORTED, UnsupportedFile, load

router = APIRouter()

@router.post("/ingest")
async def ingest(
    files: list[UploadFile] = File(...),
    access_level: str = Form("public"),
):
    level = access_control.valid_level(access_level)
    indexed = []

    for upload in files:
        ext = "." + upload.filename.rsplit(".", 1)[-1].lower()
        if ext not in SUPPORTED:
            raise HTTPException(400, f"{upload.filename}: unsupported type {ext}")
        dest = UPLOAD_DIR / upload.filename
        dest.write_bytes(await upload.read())

        try:
            sections = load(dest)
        except UnsupportedFile as e:
            raise HTTPException(400, str(e))
        chunks = chunk_sections(sections, source=upload.filename, access_level=level)
        if not chunks:
            raise HTTPException(400, f"{upload.filename}: no extractable text")
        vector_store.delete_by_source(upload.filename)  # for replacing if same file uploads
        vector_store.add_chunks(chunks)
        if ext in (".xlsx", ".xls", ".csv"):
            table_qa.register(upload.filename, dest, level)

        indexed.append({"file": upload.filename, "chunks": len(chunks)})
    bm25_index.rebuild(vector_store.all_chunks())

    return {
        "indexed": indexed,
        "access_level": level,
        "total_chunks": vector_store.count(),
    }
