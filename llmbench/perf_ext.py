#!/usr/bin/env python3
"""
Расширенные perf-тесты: batch throughput, prefill/decode разделение,
KV-cache per request, context scaling extended.

Ничего не трогает в существующем perf.py.
"""
from __future__ import annotations

import json
import statistics as st
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import requests


@dataclass
class BatchRun:
    batch_size: int
    ttft_s_mean: float
    ttft_s_p90: float
    total_time_s: float
    requests_done: int
    total_output_tokens: int
    gen_tok_s: float
    throughput_req_s: float
    errors: int


@dataclass
class BatchProfileResult:
    batch_sizes_tested: list[int]
    runs: list[BatchRun]
    optimal_batch_size: int
    max_throughput_req_s: float
    max_throughput_tok_s: float


@dataclass
class PrefillDecodeProfile:
    prefill_tok_s: float
    decode_tok_s: float
    prefill_vs_decode_ratio: float
    max_prefill_tokens: int
    max_decode_tokens: int
    kv_cache_growth_mb: Optional[float]


def run_batch_profile(
    base_url: str,
    model: str,
    batch_sizes: list[int] = None,
    prompt_len: int = 512,
    max_tokens: int = 128,
    samples_per_batch: int = 5,
    extra_body: dict = None,
) -> BatchProfileResult:
    """
    Тестирует throughput при разных batch_size.
    Отправляет samples_per_batch параллельных запросов (имитация batch).
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    headers = {"Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/chat/completions"

    runs = []
    best_throughput = 0.0
    best_batch = 1
    best_tok_s = 0.0

    for bs in batch_sizes:
        errors = 0
        total_time = 0.0
        total_tokens = 0
        ttfts = []
        done = 0

        for _ in range(samples_per_batch):
            batch_payloads = []
            for _ in range(bs):
                prompt = "Repeat this word: " + "hello " * (prompt_len // 10)
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "stream": True,
                }
                if extra_body:
                    payload.update(extra_body)
                batch_payloads.append(payload)

            start = time.monotonic()
            batch_errors = 0
            batch_tokens = 0
            batch_ttfts = []

            for payload in batch_payloads:
                try:
                    resp = requests.post(url, json=payload, headers=headers,
                                        stream=True, timeout=120)
                    start_t = None
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode("utf-8", errors="replace")
                            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                                if start_t is None:
                                    start_t = time.monotonic()
                                    batch_ttfts.append(time.monotonic() - start_t)
                                try:
                                    chunk = json.loads(decoded[6:])
                                    choices = chunk.get("choices", [{}])
                                    if choices and choices[0].get("delta", {}).get("content"):
                                        batch_tokens += 1
                                except json.JSONDecodeError:
                                    pass
                    if start_t is not None:
                        batch_ttfts.append(time.monotonic() - start_t)
                except Exception:
                    batch_errors += 1

            elapsed = time.monotonic() - start
            total_time += elapsed
            total_tokens += batch_tokens
            ttfts.extend(batch_ttfts)
            errors += batch_errors
            done += bs - batch_errors

        avg_ttft = st.mean(ttfts) if ttfts else 0.0
        p90_ttft = sorted(ttfts)[int(len(ttfts) * 0.9)] if len(ttfts) > 5 else avg_ttft
        gen_tok_s = total_tokens / total_time if total_time > 0 else 0.0
        req_s = done / total_time if total_time > 0 else 0.0

        if req_s > best_throughput:
            best_throughput = req_s
            best_batch = bs
            best_tok_s = gen_tok_s

        runs.append(BatchRun(
            batch_size=bs,
            ttft_s_mean=round(avg_ttft, 4),
            ttft_s_p90=round(p90_ttft, 4),
            total_time_s=round(total_time, 3),
            requests_done=done,
            total_output_tokens=total_tokens,
            gen_tok_s=round(gen_tok_s, 2),
            throughput_req_s=round(req_s, 2),
            errors=errors,
        ))

    return BatchProfileResult(
        batch_sizes_tested=batch_sizes,
        runs=runs,
        optimal_batch_size=best_batch,
        max_throughput_req_s=round(best_throughput, 2),
        max_throughput_tok_s=round(best_tok_s, 2),
    )


def profile_prefill_decode(
    base_url: str,
    model: str,
    prompt_lens: list[int] = None,
    max_tokens: int = 256,
) -> PrefillDecodeProfile:
    """
    Разделяет prefill и decode этапы.
    Отправляет один запрос и замеряет TTFT (prefill) vs generation rate (decode).
    """
    if prompt_lens is None:
        prompt_lens = [128, 512, 2048, 8192]
    headers = {"Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/chat/completions"

    prefill_rates = []
    decode_rates = []

    for plen in prompt_lens:
        prompt = "word " * (plen // 5)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt.strip()}],
            "max_tokens": max_tokens,
            "stream": True,
        }

        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=300)
        first_token_time = None
        token_times = []
        tokens = 0

        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8", errors="replace")
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    else:
                        token_times.append(time.monotonic())
                    tokens += 1

        if first_token_time and tokens > 1:
            prefill_rate = plen / (first_token_time - time.monotonic() + first_token_time)
            decode_rate = (tokens - 1) / (token_times[-1] - first_token_time) if token_times else 0
            prefill_rates.append(prefill_rate)
            decode_rates.append(decode_rate)

    return PrefillDecodeProfile(
        prefill_tok_s=round(st.mean(prefill_rates), 1) if prefill_rates else 0.0,
        decode_tok_s=round(st.mean(decode_rates), 1) if decode_rates else 0.0,
        prefill_vs_decode_ratio=round(
            st.mean(prefill_rates) / st.mean(decode_rates), 2
        ) if prefill_rates and decode_rates else 0.0,
        max_prefill_tokens=prompt_lens[-1] if prompt_lens else 0,
        max_decode_tokens=max_tokens,
        kv_cache_growth_mb=None,
    )

import time as _time

def _post_non_stream(url, headers, payload, timeout=120):
    import requests
    base = url.rstrip("/").replace("/chat/completions", "")
    for ep in [base + "/v1/chat/completions", base + "/v1/completions"]:
        try:
            r = requests.post(ep, headers=headers, json=dict(payload, stream=False), timeout=timeout)
            if r.status_code != 200: continue
            d = r.json()
            if "choices" in d:
                ch = d["choices"][0]
                if "message" in ch and "content" in ch["message"]: return ch["message"]["content"]
                if "text" in ch: return ch["text"]
            return None
        except Exception: continue
    return None

def _extract_text_answer(data):
    if isinstance(data, str): return data
    if isinstance(data, dict):
        if "response" in data: return data["response"]
        if "choices" in data:
            ch = data["choices"][0]
            if "message" in ch and "content" in ch["message"]: return ch["message"]["content"]
            if "text" in ch: return ch["text"]
    return None