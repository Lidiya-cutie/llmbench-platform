"""
Unified Performance Testing с Graceful Degradation

Поддерживает множество endpoint типов с автоматическим fallback:
- OpenAI-compatible streaming (vLLM, SGLang)
- Ollama (частичная совместимость)
- llama.cpp (разные версии)
- Custom endpoints

Каждый провал корректно классифицируется и сохраняется.
"""
from __future__ import annotations

import json
import time
import statistics as st
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from enum import Enum

import requests


class EndpointType(Enum):
    OPENAI_STREAMING = "openai_streaming"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


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
    endpoint_type: EndpointType = EndpointType.UNKNOWN


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
    error_summary: str = ""
    endpoint_type: EndpointType = EndpointType.UNKNOWN
    fallback_used: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d['endpoint_type'] = d['endpoint_type'].value if isinstance(d['endpoint_type'], EndpointType) else d['endpoint_type']
        return d


def _detect_endpoint_type(base_url: str, model: str, timeout: float = 10.0) -> EndpointType:
    """Автоопределение типа endpoint"""
    try:
        # Проверяем /models endpoint
        models_url = base_url.rstrip("/") + "/models"
        r = requests.get(models_url, timeout=timeout)

        if r.status_code == 200:
            data = r.json()
            # Ollama detection
            if 'ollama' in str(data).lower():
                return EndpointType.OLLAMA
            # llama.cpp detection
            if 'llama' in str(data).lower() or 'llama.cpp' in str(data).lower():
                return EndpointType.LLAMA_CPP
            return EndpointType.OPENAI_STREAMING
    except:
        pass

    # Fallback detection
    if '11434' in base_url or 'ollama' in base_url.lower():
        return EndpointType.OLLAMA
    if '8080' in base_url or 'llama' in base_url.lower():
        return EndpointType.LLAMA_CPP

    return EndpointType.UNKNOWN


def _stream_once_adaptive(base_url: str, model: str, prompt: str, max_tokens: int,
                          timeout: float, extra_body: dict, endpoint_type: EndpointType) -> OneRun:
    """Адаптивный streaming с fallback для разных endpoint типов"""

    # Fallback для Ollama (часто дает 500 на streaming)
    if endpoint_type == EndpointType.OLLAMA:
        try:
            # Пробуем non-streaming mode для Ollama
            return _non_stream_once(base_url, model, prompt, max_tokens, timeout, extra_body, endpoint_type)
        except Exception as e:
            # Если non-streaming тоже не работает, пробуем streaming с error handling
            return _stream_once_with_fallback(base_url, model, prompt, max_tokens, timeout, extra_body, endpoint_type)

    # Стандартный OpenAI-compatible streaming
    return _stream_once_with_fallback(base_url, model, prompt, max_tokens, timeout, extra_body, endpoint_type)


def _non_stream_once(base_url: str, model: str, prompt: str, max_tokens: int,
                     timeout: float, extra_body: dict, endpoint_type: EndpointType) -> OneRun:
    """Non-streaming fallback для несовместимых endpoints"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,  # Non-streaming mode
    }
    payload.update(extra_body or {})

    t0 = time.perf_counter()
    ptok = ctok = 0

    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()

        data = r.json()
        total = time.perf_counter() - t0

        # Extract tokens from response
        choices = data.get("choices", [])
        if choices and len(choices) > 0:
            content = choices[0].get("message", {}).get("content", "")
            ctok = len(content.split())  # Rough estimation

        if "usage" in data:
            ptok = data["usage"].get("prompt_tokens", 0)
            ctok = data["usage"].get("completion_tokens", ctok)

        ttft = total  # Non-streaming: TTFT ≈ total time
        decode_time = max(total - ttft, 1e-6)
        gen = ctok / decode_time if ctok else 0.0
        prefill = ptok / ttft if (ptok and ttft > 0) else 0.0

        return OneRun(True, ttft, total, ptok, ctok, gen, prefill, "", endpoint_type)

    except Exception as e:
        error_str = f"non_stream_{str(e)[:100]}"
        return OneRun(False, 0, 0, 0, 0, 0, 0, error_str, endpoint_type)


def _stream_once_with_fallback(base_url: str, model: str, prompt: str, max_tokens: int,
                                 timeout: float, extra_body: dict, endpoint_type: EndpointType) -> OneRun:
    """Стандартный streaming с улучшенным error handling"""
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

                try:
                    obj = json.loads(data)
                    choices = obj.get("choices") or []
                    if ttft is None and choices and choices[0].get("delta", {}).get("content"):
                        ttft = time.perf_counter() - t0
                    if obj.get("usage"):
                        ptok = obj["usage"].get("prompt_tokens", ptok)
                        ctok = obj["usage"].get("completion_tokens", ctok)
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON lines

        total = time.perf_counter() - t0
        ttft = ttft if ttft is not None else total
        decode_time = max(total - ttft, 1e-6)
        gen = ctok / decode_time if ctok else 0.0
        prefill = ptok / ttft if (ptok and ttft > 0) else 0.0

        return OneRun(True, ttft, total, ptok, ctok, gen, prefill, "", endpoint_type)

    except requests.exceptions.HTTPError as e:
        # Specialized error classification
        if e.response.status_code == 500:
            return OneRun(False, 0, 0, 0, 0, 0, 0, "500_internal_server_error", endpoint_type)
        elif e.response.status_code == 503:
            return OneRun(False, 0, 0, 0, 0, 0, 0, "503_service_unavailable", endpoint_type)
        elif e.response.status_code == 404:
            return OneRun(False, 0, 0, 0, 0, 0, 0, "404_model_not_found", endpoint_type)
        else:
            return OneRun(False, 0, 0, 0, 0, 0, 0, f"http_{e.response.status_code}", endpoint_type)

    except requests.exceptions.Timeout:
        return OneRun(False, 0, 0, 0, 0, 0, 0, "timeout", endpoint_type)
    except requests.exceptions.ConnectionError:
        return OneRun(False, 0, 0, 0, 0, 0, 0, "connection_failed", endpoint_type)
    except Exception as e:
        error_str = str(e)[:100]
        if "stream" in error_str.lower():
            error_str = "streaming_error"
        return OneRun(False, 0, 0, 0, 0, 0, 0, error_str, endpoint_type)


def _pctl(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def aggregate_adaptive(runs: List[OneRun], prompt_len: int, max_tokens: int,
                       endpoint_type: EndpointType, fallback_used: bool) -> PerfResult:
    """Умная агрегация с детальной диагностикой"""

    good = [r for r in runs if r.ok]
    bad = [r for r in runs if not r.ok]

    ttft = [r.ttft_s for r in good]
    lat = [r.total_s for r in good]
    gen = [r.gen_tok_s for r in good if r.gen_tok_s > 0]
    pre = [r.prefill_tok_s for r in good if r.prefill_tok_s > 0]

    # Detailed error analysis
    error_summary = ""
    if bad:
        error_types: Dict[str, int] = {}
        for r in bad:
            # Классифицируем ошибки
            error_key = r.error.split(':')[0] if ':' in r.error else r.error[:20]
            error_types[error_key] = error_types.get(error_key, 0) + 1

        error_parts = []
        for error_type, count in sorted(error_types.items()):
            error_parts.append(f"{error_type}={count}")
        error_summary = ", ".join(error_parts)

    return PerfResult(
        n=len(runs),
        ok=len(good),
        ttft_s_mean=round(st.mean(ttft), 4) if ttft else 0.0,
        ttft_s_p90=round(_pctl(ttft, 0.9), 4),
        latency_s_p50=round(_pctl(lat, 0.5), 4),
        latency_s_p90=round(_pctl(lat, 0.9), 4),
        latency_s_p99=round(_pctl(lat, 0.99), 4),
        gen_tok_s_mean=round(st.mean(gen), 2) if gen else 0.0,
        gen_tok_s_median=round(st.median(gen), 2) if gen else 0.0,
        prefill_tok_s_mean=round(st.mean(pre), 2) if pre else 0.0,
        prompt_len_tokens=prompt_len,
        max_tokens=max_tokens,
        error_summary=error_summary,
        endpoint_type=endpoint_type,
        fallback_used=fallback_used,
    )


def run_perf_unified(base_url: str, model: str, *, n: int = 12, max_tokens: int = 256,
                     prompt: str = "Опиши в трёх абзацах, как работает механизм внимания в трансформерах.",
                     warmup: int = 1, timeout: float = 600.0, extra_body: dict = None,
                     auto_detect: bool = True, force_fallback: bool = False) -> PerfResult:
    """
    Unified perf тест с автоматическим fallback и детальной диагностикой

    Args:
        auto_detect: автоматически определять тип endpoint
        force_fallback: использовать non-streaming mode
    """

    # Определяем тип endpoint
    endpoint_type = _detect_endpoint_type(base_url, model) if auto_detect else EndpointType.UNKNOWN

    print(f"🔍 Detected endpoint: {endpoint_type.value}")

    # Warmup (короткий тест для проверки)
    if warmup > 0:
        try:
            if force_fallback or endpoint_type == EndpointType.OLLAMA:
                _non_stream_once(base_url, model, prompt, 16, timeout, extra_body, endpoint_type)
            else:
                _stream_once_with_fallback(base_url, model, prompt, 16, timeout, extra_body, endpoint_type)
        except:
            pass  # Warmup failures are OK

    # Основной тест
    runs = []
    fallback_used = force_fallback

    for i in range(n):
        if force_fallback or (fallback_used and endpoint_type == EndpointType.OLLAMA):
            # Используем non-streaming для проблемных endpoints
            run = _non_stream_once(base_url, model, prompt, max_tokens, timeout, extra_body or {}, endpoint_type)
            runs.append(run)

            # Если первый non-streaming успешен, продолжаем
            if run.ok:
                continue
            # Если провалился, пробуем streaming
            fallback_used = False
        else:
            run = _stream_once_adaptive(base_url, model, prompt, max_tokens, timeout, extra_body or {}, endpoint_type)
            runs.append(run)

            # Если streaming провалился, переключаем на fallback для остальных
            if not run.ok and i == 0 and "500" in run.error:
                print(f"⚠️ Streaming failed, switching to non-streaming fallback...")
                fallback_used = True

    approx_prompt_tokens = next((r.prompt_tokens for r in runs if r.ok and r.prompt_tokens), len(prompt) // 3)

    result = aggregate_adaptive(runs, approx_prompt_tokens, max_tokens, endpoint_type, fallback_used)

    print(f"📊 Result: n={result.n}, ok={result.ok}, endpoint={result.endpoint_type.value}")
    if result.error_summary:
        print(f"   Errors: {result.error_summary}")
    if result.fallback_used:
        print(f"   ✅ Fallback mode used (non-streaming)")

    return result


# Backwards compatibility alias
run_perf = run_perf_unified
