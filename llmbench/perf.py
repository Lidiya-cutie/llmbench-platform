"""
Performance benchmark против OpenAI-совместимого эндпоинта.

Меряем то, что реально важно для локального инференса на 4090:
  - TTFT (time to first token) через streaming;
  - generation tokens/sec (decode);
  - prompt processing tokens/sec (prefill, приближённо: prompt_tokens / TTFT);
  - end-to-end latency percentiles (p50/p90/p99);
  - context scaling: те же метрики на разных длинах промпта.

Работает с llama.cpp (llama-server), vLLM, SGLang, Ollama — любым, кто отдаёт
/v1/chat/completions со streaming. Именно так запускается локальная 27B-модель
(./scripts/start_llama_server.sh -> :8080).
"""
from __future__ import annotations

import json
import statistics as st
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests


@dataclass
class OneRun:
    ok: bool
    ttft_s: float
    total_s: float
    prompt_tokens: int
    completion_tokens: int
    gen_tok_s: float
    prefill_tok_s: float
    error: str = ""


@dataclass
class PerfResult:
    n: int
    ok: int
    ttft_s_mean: float
    ttft_s_p90: float
    latency_s_p50: float
    latency_s_p90: float
    latency_s_p99: float
    gen_tok_s_mean: float
    gen_tok_s_median: float
    prefill_tok_s_mean: float
    prompt_len_tokens: int
    max_tokens: int
    error_summary: str = ""  # Добавляем сводку ошибок

    def to_dict(self) -> dict:
        return asdict(self)


def _pctl(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _stream_once(base_url: str, model: str, prompt: str, max_tokens: int,
                 timeout: float, extra_body: dict) -> OneRun:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload.update(extra_body or {})
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    ptok = ctok = 0
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                choices = obj.get("choices") or []
                if ttft is None and choices and choices[0].get("delta", {}).get("content"):
                    ttft = time.perf_counter() - t0
                if obj.get("usage"):
                    ptok = obj["usage"].get("prompt_tokens", ptok)
                    ctok = obj["usage"].get("completion_tokens", ctok)
        total = time.perf_counter() - t0
        ttft = ttft if ttft is not None else total
        decode_time = max(total - ttft, 1e-6)
        gen = ctok / decode_time if ctok else 0.0
        prefill = ptok / ttft if (ptok and ttft > 0) else 0.0
        return OneRun(True, ttft, total, ptok, ctok, gen, prefill)
    except requests.exceptions.HTTPError as e:
        # Special handling for incompatible endpoints (Ollama 500 errors, etc.)
        error_msg = f"HTTP_{e.response.status_code}"
        if e.response.status_code == 500:
            error_msg = "500_incompatible_streaming_api"
        elif e.response.status_code == 404:
            error_msg = "404_model_not_found"
        elif e.response.status_code == 401:
            error_msg = "401_unauthorized"
        return OneRun(False, 0, 0, 0, 0, 0, 0, error_msg)
    except requests.exceptions.Timeout:
        return OneRun(False, 0, 0, 0, 0, 0, 0, "timeout")
    except requests.exceptions.ConnectionError:
        return OneRun(False, 0, 0, 0, 0, 0, 0, "connection_failed")
    except json.JSONDecodeError as e:
        return OneRun(False, 0, 0, 0, 0, 0, 0, f"invalid_json_streaming")
    except Exception as e:
        # Generic error with truncated message
        error_str = str(e)[:200]
        # Classify common errors
        if "stream" in error_str.lower():
            error_str = "streaming_error: " + error_str[:150]
        elif "timeout" in error_str.lower():
            error_str = "timeout_error: " + error_str[:150]
        elif "connection" in error_str.lower():
            error_str = "connection_error: " + error_str[:150]
        return OneRun(False, 0, 0, 0, 0, 0, 0, error_str)


def aggregate(runs: List[OneRun], prompt_len: int, max_tokens: int) -> PerfResult:
    good = [r for r in runs if r.ok]
    bad = [r for r in runs if not r.ok]

    ttft = [r.ttft_s for r in good]
    lat = [r.total_s for r in good]
    gen = [r.gen_tok_s for r in good if r.gen_tok_s > 0]
    pre = [r.prefill_tok_s for r in good if r.prefill_tok_s > 0]

    # Собираем информацию об ошибках для диагностики
    error_summary = ""
    if bad:
        error_types = {}
        for r in bad:
            error_type = r.error.split(':')[0] if ':' in r.error else r.error[:20]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        error_parts = []
        for error_type, count in error_types.items():
            error_parts.append(f"{error_type}={count}")
        error_summary = ", ".join(error_parts)

    return PerfResult(
        n=len(runs), ok=len(good),
        ttft_s_mean=round(st.mean(ttft), 4) if ttft else 0.0,
        ttft_s_p90=round(_pctl(ttft, 0.9), 4),
        latency_s_p50=round(_pctl(lat, 0.5), 4),
        latency_s_p90=round(_pctl(lat, 0.9), 4),
        latency_s_p99=round(_pctl(lat, 0.99), 4),
        gen_tok_s_mean=round(st.mean(gen), 2) if gen else 0.0,
        gen_tok_s_median=round(st.median(gen), 2) if gen else 0.0,
        prefill_tok_s_mean=round(st.mean(pre), 2) if pre else 0.0,
        prompt_len_tokens=prompt_len, max_tokens=max_tokens,
        error_summary=error_summary,
    )


def run_perf(base_url: str, model: str, *, n: int = 8, max_tokens: int = 256,
             prompt: str = "Опиши в трёх абзацах, как работает механизм внимания в трансформерах.",
             warmup: int = 1, timeout: float = 600.0, extra_body: dict = None) -> PerfResult:
    for _ in range(max(warmup, 0)):
        _stream_once(base_url, model, prompt, 16, timeout, extra_body)
    runs = [_stream_once(base_url, model, prompt, max_tokens, timeout, extra_body) for _ in range(n)]
    approx_prompt_tokens = next((r.prompt_tokens for r in runs if r.ok and r.prompt_tokens), len(prompt) // 3)
    return aggregate(runs, approx_prompt_tokens, max_tokens)


def run_context_scaling(base_url: str, model: str, *, lengths_tokens=(1000, 8000, 32000),
                        max_tokens: int = 128, extra_body: dict = None) -> List[PerfResult]:
    """Прогон на нескольких длинах контекста. Наполнитель — повторяемый текст."""
    filler = "Контекстное окно проверяется длинным повторяющимся текстом. "
    results = []
    for L in lengths_tokens:
        prompt = (filler * (L // 8))[: L * 4] + "\n\nКоротко ответь: сколько будет 17*23?"
        results.append(run_perf(base_url, model, n=3, max_tokens=max_tokens,
                                 prompt=prompt, extra_body=extra_body))
    return results
