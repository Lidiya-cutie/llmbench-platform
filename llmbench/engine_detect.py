"""
Авто-детект движка инференса и генерация launch_cmd.

Позволяет НЕ указывать `engine` и `launch_cmd` в models.yaml -- достаточно
`repo` и `quant_format`. Модуль сам определяет, какой движок подходит,
и генерирует команду запуска через предопределённые шаблоны.

Поддерживаемые движки:
  - vLLM (рекомендуемый для BF16/FP16, AWQ, GPTQ)
  - SGLang (альтернатива vLLM, быстрее prefill)
  - llama.cpp (GGUF, QAT-кванты, кастомные кернелы)
  - HuggingFace TGI (текстовые модели с HF)
  - Ollama (GGUF, модельный реестр)
  - TensorRT-LLM (TRT-оптимизированные)
  - ExLlamaV2 (GPTQ/LoRA)

Никаких изменений в существующие файлы не вносит.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    engine: str
    launch_cmd: str
    served_model_name: str
    confidence: float  # 0.0 -- 1.0
    warnings: list[str] = field(default_factory=list)


# --- Детекторы ---------------------------------------------------------


def _detect_gguf(quant_format: str) -> float:
    q = quant_format.lower()
    if any(x in q for x in ("gguf", "q1_", "q2_", "q3_", "q4_", "q5_", "q6_", "q8_")):
        return 0.95
    if "ternary" in q or "1.58" in q or "1.125" in q:
        return 0.90
    return 0.0


def _detect_gptq(quant_format: str) -> float:
    q = quant_format.lower()
    if "gptq" in q or "awq" in q:
        return 0.95
    return 0.0


def _detect_exl2(quant_format: str) -> float:
    q = quant_format.lower()
    if any(x in q for x in ("exl2", "exllama", "gptq-lora")):
        return 0.90
    return 0.0


def _detect_trt(quant_format: str) -> float:
    q = quant_format.lower()
    if any(x in q for x in ("trt-llm", "tensorrt", "trtllm", "engine")):
        return 0.95
    return 0.0


# --- Генераторы launch_cmd -----------------------------------------------


def _cmd_vllm(repo: str, port: int = 8077, gpu_index: int = 0, **kwargs) -> str:
    dtype = kwargs.get("dtype", "bfloat16")
    max_len = kwargs.get("context_max", 8192)
    gpu_util = kwargs.get("gpu_memory_utilization", 0.90)
    model_name = kwargs.get("served_model_name", repo.split("/")[-1])
    extra = kwargs.get("extra_args", "")
    return (
        f"docker run -d --name vllm-{model_name} --gpus device={gpu_index} "
        f"-v /opt/llm-data/.cache_huggingface:/root/.cache/huggingface "
        f"-p {port}:{port} "
        f"mlops-triton-vllm:24.12-qwen3 "
        f"vllm serve /root/.cache/huggingface/{repo} "
        f"--host 0.0.0.0 --port {port} --dtype {dtype} "
        f"--max-model-len {max_len} --gpu-memory-utilization {gpu_util} "
        f"--served-model-name {model_name} "
        f"--trust-remote-code {extra}"
    ).strip()


def _cmd_sglang(repo: str, port: int = 8078, gpu_index: int = 0, **kwargs) -> str:
    dtype = kwargs.get("dtype", "bfloat16")
    max_len = kwargs.get("context_max", 8192)
    model_name = kwargs.get("served_model_name", repo.split("/")[-1])
    return (
        f"docker run -d --name sgl-{model_name} --gpus device={gpu_index} "
        f"-v /opt/llm-data/.cache_huggingface:/root/.cache/huggingface "
        f"-p {port}:{port} "
        f"sglang/sglang:latest "
        f"python3 -m sglang.launch_server --model {repo} "
        f"--host 0.0.0.0 --port {port} --dtype {dtype} "
        f"--max-model-len {max_len} "
        f"--trust-remote-code"
    ).strip()


def _cmd_llamacpp(repo: str, port: int = 8080, gpu_index: int = 0, **kwargs) -> str:
    gguf_path = kwargs.get("gguf_path", f"/opt/llm-data/models/{repo.split('/')[-1]}")
    model_name = kwargs.get("served_model_name", "local-27b")
    extra_env = kwargs.get("env", "")
    extra_args = kwargs.get("extra_args", "--no-mmap --numa")
    env_prefix = f"{extra_env} " if extra_env else ""
    return (
        f"{env_prefix}./build/bin/llama-server "
        f"-m {gguf_path} "
        f"--host 0.0.0.0 --port {port} "
        f"-ngl {kwargs.get('ngl', 32)} "
        f"-c {kwargs.get('context_max', 8192)} "
        f"--cont-batching "
        f"{extra_args} "
        f"--port {port}"
    ).strip()


def _cmd_tgi(repo: str, port: int = 8079, gpu_index: int = 0, **kwargs) -> str:
    dtype = kwargs.get("dtype", "bfloat16")
    max_len = kwargs.get("context_max", 4096)
    model_name = kwargs.get("served_model_name", repo.split("/")[-1])
    return (
        f"docker run -d --name tgi-{model_name} --gpus device={gpu_index} "
        f"-v /opt/llm-data/.cache_huggingface:/root/.cache/huggingface "
        f"-p {port}:80 "
        f"ghcr.io/huggingface/text-generation-inference:latest "
        f"--model-id {repo} --dtype {dtype} "
        f"--max-total-tokens {max_len} "
        f"--trust-remote-code"
    ).strip()


def _cmd_ollama(repo: str, port: int = 8081, **kwargs) -> str:
    model_name = kwargs.get("served_model_name", repo.split("/")[-1])
    return (
        f"ollama run {model_name} "
        f"&& OLLAMA_HOST=0.0.0.0:{port} ollama serve"
    ).strip()


def _cmd_trtllm(repo: str, port: int = 8082, gpu_index: int = 0, **kwargs) -> str:
    model_name = kwargs.get("served_model_name", repo.split("/")[-1])
    engine_dir = kwargs.get("engine_dir", f"/opt/llm-data/trt-engines/{model_name}")
    return (
        f"docker run -d --name trt-{model_name} --gpus device={gpu_index} "
        f"-v {engine_dir}:/engines "
        f"-p {port}:{port} "
        f"nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3 "
        f"python3 /opt/tensorrtllm/scripts/launch_triton_server.py "
        f"--model_repo /engines --world_size 1"
    ).strip()


# --- Карта: движок -> (детектор, генератор) -------------------------------


_ENGINE_MAP = {
    "vLLM": (_detect_gptq, _cmd_vllm),
    "SGLang": (lambda q: 0.0, _cmd_sglang),
    "llama.cpp": (_detect_gguf, _cmd_llamacpp),
    "HuggingFace TGI": (lambda q: 0.0, _cmd_tgi),
    "Ollama": (_detect_gguf, _cmd_ollama),
    "TensorRT-LLM": (_detect_trt, _cmd_trtllm),
}


def detect_engine(model_cfg: dict) -> EngineConfig:
    explicit_engine = model_cfg.get("engine", "").lower()
    quant = model_cfg.get("quant_format", "")
    repo = model_cfg.get("repo", "")
    hw = model_cfg.get("hardware", {})
    gpu_index = hw.get("gpu_index", 0)
    served_name = model_cfg.get("served_model_name",
                                repo.split("/")[-1] if repo else "model")

    kwargs = {k: v for k, v in model_cfg.items()
              if k not in ["engine", "repo", "quant_format", "hardware", "served_model_name"]}

    # Явно указанный движок
    if explicit_engine:
        for engine_name, (detector, generator) in _ENGINE_MAP.items():
            if explicit_engine in engine_name.lower():
                cmd = generator(repo, gpu_index=gpu_index,
                                served_model_name=served_name, **kwargs)
                return EngineConfig(
                    engine=engine_name,
                    launch_cmd=cmd,
                    served_model_name=served_name,
                    confidence=1.0,
                )
        return EngineConfig(
            engine=explicit_engine,
            launch_cmd="",
            served_model_name=served_name,
            confidence=0.5,
            warnings=[f"Неизвестный движок '{explicit_engine}'. Укажите launch_cmd вручную."],
        )

    # Авто-детект по quant_format
    candidates = []
    for engine_name, (detector, generator) in _ENGINE_MAP.items():
        score = detector(quant)
        if score > 0.0:
            cmd = generator(repo, gpu_index=gpu_index,
                            served_model_name=served_name, **kwargs)
            candidates.append((score, engine_name, cmd))

    if not candidates:
        cmd = _cmd_vllm(repo, gpu_index=gpu_index,
                        served_model_name=served_name, **kwargs)
        candidates.append((0.5, "vLLM (fallback)", cmd))

    candidates.sort(reverse=True)
    best = candidates[0]
    warnings = []
    if best[0] < 0.8:
        warnings.append(f"Низкая уверенность ({best[0]:.0%}). Проверьте launch_cmd.")

    return EngineConfig(
        engine=best[1],
        launch_cmd=best[2],
        served_model_name=served_name,
        confidence=best[0],
        warnings=warnings,
    )


def validate_engine_config(cfg: EngineConfig) -> list[str]:
    warnings = []
    if not cfg.launch_cmd:
        warnings.append("launch_cmd пустой -- модель не запустится.")
    if cfg.confidence < 0.5:
        warnings.append("Низкая уверенность в типе движка. Укажите engine явно.")
    if not cfg.served_model_name:
        warnings.append("served_model_name не задан.")
    return warnings
