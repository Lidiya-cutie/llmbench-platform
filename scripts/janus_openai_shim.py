#!/usr/bin/env python3
"""Minimal OpenAI-compatible text chat server for Janus-Pro (GPU via CUDA_VISIBLE_DEVICES)."""
from __future__ import annotations

import collections
import collections.abc
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

for _n in ("Mapping", "MutableMapping", "Sequence", "Iterable"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

import torch
from transformers import AutoModelForCausalLM

JANUS_ROOT = Path("/opt/llm-data/my_train_models/deepseek_and_colpali/Janus")
sys.path.insert(0, str(JANUS_ROOT))

from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: E402

_LOCK = threading.Lock()

MODEL_PATH = os.environ.get(
    "JANUS_MODEL_PATH",
    "/opt/llm-data/my_train_models/deepseek_and_colpali/Janus/model2",
)
SERVED = os.environ.get("JANUS_SERVED_NAME", "Janus-Pro-7B")
PORT = int(os.environ.get("JANUS_PORT", "8072"))
MAX_NEW = int(os.environ.get("JANUS_MAX_NEW", "1024"))

print(f"[janus-shim] loading {MODEL_PATH} as {SERVED}", flush=True)
processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer
model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
)
model = model.to(torch.bfloat16).cuda().eval()
print("[janus-shim] ready", flush=True)


@torch.inference_mode()
def chat_text(messages: list[dict], max_tokens: int, temperature: float) -> str:
    conversation = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(x.get("text", "")) for x in content if isinstance(x, dict)
            )
        if role == "system":
            continue
        conversation.append(
            {
                "role": "User" if role == "user" else "Assistant",
                "content": str(content),
            }
        )
    if not conversation or conversation[-1]["role"] != "Assistant":
        conversation.append({"role": "Assistant", "content": ""})

    sft = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    input_ids = tokenizer.encode(sft, return_tensors="pt").cuda()
    do_sample = float(temperature or 0) > 0
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max(1, int(max_tokens or MAX_NEW)),
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
    with _LOCK:
        out = model.language_model.generate(**gen_kwargs)
    gen = out[0, input_ids.shape[-1] :]
    return tokenizer.decode(gen.tolist(), skip_special_tokens=True)


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
                            "owned_by": "janus-shim",
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
                "id": f"chatcmpl-janus-{int(time.time())}",
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
    print(f"[janus-shim] listening :{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
