from pathlib import Path
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

def _read_env_file():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())

_read_env_file()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_PATH = CHROMA_DIR / "bm25.pkl"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e2b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))
RRF_K = 60
COLLECTION_NAME = "documents"
MEMORY_COLLECTION = "chat_memory"
SHORT_TERM_TURNS = 6 
LONG_TERM_TOP_K = 3
LONG_TERM_MIN_SCORE = 0.4 
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))
CACHE_MAX_ENTRIES = 256
ROLE_LEVELS = {
    "public": 0,
    "employee": 1,
    "manager": 2,
}
DEFAULT_ROLE = "public"
