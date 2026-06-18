import json
import requests
from config import LLM_MODEL, OLLAMA_HOST
class LLMError(Exception):
    pass
def generate(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise LLMError(f"Ollama call failed - is it running? ({e})") from e
    return resp.json().get("response", "").strip()

def generate_json(prompt: str, system: str | None = None) -> dict:
    raw = generate(prompt, system=system, temperature=0.0)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
