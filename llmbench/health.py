"""
Health-check для OpenAI-совместимых эндпоинтов локальных LLM.

Проверяет готовность модели к инференсу: HTTP-доступность, наличие модели в
/v1/models, базовый generate-тест. Работает с vLLM, SGLang, llama.cpp, Ollama,
TGI, Triton — любым OpenAI-совместимым сервером.

Не трогает запущенные процессы — только HTTP-проверки.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


@dataclass
class HealthStatus:
    ok: bool
    base_url: str
    model_name: Optional[str]
    error: Optional[str]
    latency_ms: Optional[float]
    models_list: list[str] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "models_count": len(self.models_list),
            "models": self.models_list[:5],
        }


def _strip_v1(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url[:-3]
    return url


def check_openai_endpoint(base_url: str, model_name: str = None, timeout: int = 30) -> HealthStatus:
    if not _HAS_REQUESTS:
        return HealthStatus(
            ok=False,
            base_url=base_url,
            model_name=model_name,
            error="requests module not installed",
            latency_ms=None,
        )

    base = _strip_v1(base_url)
    models_url = f"{base}/v1/models"
    t0 = time.monotonic()

    try:
        resp = requests.get(models_url, timeout=timeout)
        latency_ms = (time.monotonic() - t0) * 1000

        if resp.status_code != 200:
            return HealthStatus(
                ok=False,
                base_url=base_url,
                model_name=model_name,
                error=f"HTTP {resp.status_code}",
                latency_ms=round(latency_ms, 1),
            )

        data = resp.json()
        models = []
        if isinstance(data, dict):
            if "object" in data and data["object"] == "list":
                models = [m.get("id", "") for m in data.get("data", [])]
            elif "data" in data:
                models = [m.get("id", "") for m in data["data"]]
        elif isinstance(data, list):
            models = [m.get("id", "") for m in data]

        if model_name:
            model_found = any(model_name in m for m in models)
            if not model_found:
                return HealthStatus(
                    ok=False,
                    base_url=base_url,
                    model_name=model_name,
                    error=f"Model '{model_name}' not found in /v1/models",
                    latency_ms=round(latency_ms, 1),
                    models_list=models,
                )

        return HealthStatus(
            ok=True,
            base_url=base_url,
            model_name=model_name,
            error=None,
            latency_ms=round(latency_ms, 1),
            models_list=models,
            raw_response=data,
        )

    except requests.exceptions.Timeout:
        return HealthStatus(
            ok=False,
            base_url=base_url,
            model_name=model_name,
            error=f"Timeout after {timeout}s",
            latency_ms=None,
        )
    except requests.exceptions.ConnectionError as e:
        return HealthStatus(
            ok=False,
            base_url=base_url,
            model_name=model_name,
            error=f"Connection refused: {e}",
            latency_ms=None,
        )
    except Exception as e:
        return HealthStatus(
            ok=False,
            base_url=base_url,
            model_name=model_name,
            error=str(e),
            latency_ms=None,
        )


def check_model_loaded(base_url: str, model_name: str, timeout: int = 30) -> bool:
    status = check_openai_endpoint(base_url, model_name=model_name, timeout=timeout)
    return status.ok


def wait_ready(base_url: str, model_name: str = None, timeout: int = 180, interval: int = 5) -> HealthStatus:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        status = check_openai_endpoint(base_url, model_name=model_name, timeout=min(10, timeout))

        if status.ok:
            return status

        if status.error and "Timeout" not in status.error and "Connection" not in status.error:
            return status

        wait = min(interval, deadline - time.monotonic())
        if wait <= 0:
            break
        time.sleep(wait)

    return HealthStatus(
        ok=False,
        base_url=base_url,
        model_name=model_name,
        error=f"Timeout after {timeout}s waiting for {base_url}",
        latency_ms=None,
    )


def check_generate(base_url: str, model_name: str = None, prompt: str = "2+2=", timeout: int = 60) -> Dict[str, Any]:
    if not _HAS_REQUESTS:
        return {"ok": False, "error": "requests module not installed"}

    base = _strip_v1(base_url)
    url = f"{base}/v1/chat/completions"

    payload = {
        "model": model_name or "model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    try:
        t0 = time.monotonic()
        resp = requests.post(url, json=payload, timeout=timeout)
        latency_ms = (time.monotonic() - t0) * 1000

        if resp.status_code == 200:
            data = resp.json()
            return {
                "ok": True,
                "latency_ms": round(latency_ms, 1),
                "text": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
            }

        return {
            "ok": False,
            "error": f"HTTP {resp.status_code}",
            "latency_ms": round(latency_ms, 1),
        }

    except requests.exceptions.Timeout:
        return {"ok": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
