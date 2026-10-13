#!/usr/bin/env python3
"""Minimal OpenAI-compatible text chat server for InternVL3.5-HF (transformers)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = os.environ.get(
    "INTERNVL_MODEL_PATH",
    "/opt/llm-data/.cache_huggingface/hub/models--OpenGVLab--InternVL3_5-8B-HF/snapshots/741a7d03020411e666c6109218ab71e08151ef86",
)
SERVED = os.environ.get("INTERNVL_SERVED_NAME", "InternVL3_5-8B-HF")
PORT = int(os.environ.get("INTERNVL_PORT", "8071"))
MAX_NEW = int(os.environ.get("INTERNVL_MAX_NEW", "1024"))

_LOCK = threading.Lock()

print(f"[internvl-shim] loading {MODEL_PATH} as {SERVED}", flush=True)
processor = AutoProcessor.from_pretrained(
    MODEL_PATH, local_files_only=True, trust_remote_code=True
)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = model.to("cuda").eval()
print("[internvl-shim] ready", flush=True)


def _plain_content(content) -> str:
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict):
                parts.append(str(x.get("text", "")))
            else:
                parts.append(str(x))
        return " ".join(p for p in parts if p)
    return str(content or "")


@torch.inference_mode()
def chat_text(messages: list[dict], max_tokens: int, temperature: float) -> str:
    conv = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            text = _plain_content(m.get("content"))
            if text:
                conv.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": text}],
                    }
                )
            continue
        role_out = "user" if role == "user" else "assistant"
        conv.append(
            {
                "role": role_out,
                "content": [{"type": "text", "text": _plain_content(m.get("content"))}],
            }
        )
    if not conv:
        conv = [{"role": "user", "content": [{"type": "text", "text": ""}]}]

    inputs = processor.apply_chat_template(
        conv,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    do_sample = float(temperature or 0) > 0
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max(1, int(max_tokens or MAX_NEW)),
        do_sample=do_sample,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)

    with _LOCK:
        out = model.generate(**gen_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    gen = out[0, prompt_len:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, code: int, obj: dict):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/v1/models") or self.path.startswith("/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": SERVED,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "internvl-shim",
                        }
                    ],
                },
            )
            return
        if self.path.startswith("/health"):
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._json(400, {"error": "bad json"})
            return

        if not (
            self.path.startswith("/v1/chat/completions")
            or self.path.startswith("/chat/completions")
        ):
            self._json(404, {"error": f"no route {self.path}"})
            return

        messages = payload.get("messages") or []
        max_tokens = int(payload.get("max_tokens") or MAX_NEW)
        temperature = float(payload.get("temperature") or 0)
        try:
            text = chat_text(messages, max_tokens, temperature)
        except Exception as e:
            self._json(500, {"error": {"message": str(e), "type": type(e).__name__}})
            return

        self._json(
            200,
            {
                "id": f"chatcmpl-internvl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model") or SERVED,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        )


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[internvl-shim] listening :{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
