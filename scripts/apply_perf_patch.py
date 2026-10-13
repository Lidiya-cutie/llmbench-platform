import shutil, re
from pathlib import Path

SCRIPTS = Path("/opt/llm-data/llmbench/scripts")
perf = SCRIPTS / "perf_ext.py"

bak = perf.with_suffix(".py.bak-perfpatch")
if not bak.exists():
    shutil.copy2(perf, bak)
    print(f"Backup: {bak.name}")
else:
    print(f"Backup exists: {bak.name}")

src = perf.read_text()

if "def _post_non_stream" in src:
    print("Already patched, skipping.")
    exit(0)

patch = '''

# --- perfpatch v1: fallback /chat/completions -> /completions, non-stream, text answers ---
import time as _time

def _post_non_stream(url, headers, payload, timeout=120):
    """Non-streaming fallback: try /v1/chat/completions then /v1/completions."""
    import requests
    base = url.rstrip("/").replace("/chat/completions", "")
    for ep in [f"{base}/v1/chat/completions", f"{base}/v1/completions"]:
        try:
            payload_ns = dict(payload, stream=False)
            r = requests.post(ep, headers=headers, json=payload_ns, timeout=timeout)
            if r.status_code != 200:
                continue
            data = r.json()
            if "choices" in data:
                ch = data["choices"][0]
                if "message" in ch and "content" in ch["message"]:
                    return ch["message"]["content"]
                if "text" in ch:
                    return ch["text"]
            return None
        except Exception:
            continue
    return None

def _extract_text_answer(data):
    """Extract text from Ollama-style response."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if "response" in data:
            return data["response"]
        if "choices" in data:
            ch = data["choices"][0]
            if "message" in ch and "content" in ch["message"]:
                return ch["message"]["content"]
            if "text" in ch:
                return ch["text"]
    return None
'''

perf.write_text(src + patch)
print(f"Patched: {perf.name}")
