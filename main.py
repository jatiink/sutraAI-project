from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from config import BASE_DIR
from routes import chat, ingest

app = FastAPI(title="Local Grounded RAG Chatbot")

app.include_router(ingest.router, tags=["ingest"])
app.include_router(chat.router, tags=["chat"])
static_dir = BASE_DIR / "static"

@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)