#!/usr/bin/env python3
"""Запуск code benchmarks только для доступных моделей.

Автоматическая проверка API + запуск EvalPlus для готовых моделей.
"""
from __future__ import annotations

import subprocess
import time
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

LLMBENCH = Path("/opt/llm-data/llmbench")
SCRIPTS = LLMBENCH / "scripts"
LOGS = LLMBENCH / "logs"
OUT = LLMBENCH / "out"

# Конфигурация моделей для проверки
MODELS_TO_CHECK = {
    "local-27b-27b-ternary": {
        "api_url": "http://127.0.0.1:8082/v1",
        "model_endpoint": "http://127.0.0.1:8082/v1/models",
        "evalplus_model": "Ternary-local-27b-Q2_0.gguf",
        "backend": "ollama",
    },
    "gemma-3-27b-awq": {
        "api_url": "http://127.0.0.1:19892/v1",
        "model_endpoint": "http://127.0.0.1:19892/v1/models",
        "evalplus_model": "gemma-3-27b-awq",
        "backend": "vllm",
    },
    "ollama-qwen3-8b": {
        "api_url": "http://127.0.0.1:11434/v1",
        "model_endpoint": "http://127.0.0.1:11434/api/tags",
        "evalplus_model": "qwen3:8b",
        "backend": "ollama",
    },
    "deepseek-v4-flash": {
        "api_url": "http://31.28.10.135:8000/v1",
        "model_endpoint": "http://31.28.10.135:8000/v1/models",
        "evalplus_model": "deepseek-v4-flash-dspark",
        "backend": "openai",
    },
}


def check_api_ready(model_config: dict) -> bool:
    """Проверка доступности API модели."""
    try:
        resp = requests.get(model_config["model_endpoint"], timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_available_models() -> Dict[str, dict]:
    """Проверка доступности моделей."""
    available = {}

    for model_name, config in MODELS_TO_CHECK.items():
        print(f"Checking {model_name}...")

        if check_api_ready(config):
            print(f"  ✅ Available")
            available[model_name] = config
        else:
            print(f"  ❌ Not available")

    return available


def launch_code_benchmark(model_name: str, model_config: dict, benchmark: str = "humaneval+"):
    """Запуск code benchmark для модели."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS / f"codebench_{model_name}_{benchmark}_{timestamp}.log"

    cmd = [
        str(SCRIPTS / "run_evalplus_codebench.py"),
        "--model", model_name,
        "--benchmark", benchmark,
        "--llmbench", str(LLMBENCH),
        "--update-card",
    ]

    print(f"🚀 Launching {benchmark} for {model_name}")
    print(f"   Log: {log_file}")

    # Запуск в фоне с логированием
    with open(log_file, "w") as log:
        subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=LLMBENCH,
        )

    return log_file


def main():
    print("=== AVAILABLE CODE BENCHMARK MODELS ===")
    print(f"Timestamp: {datetime.now().isoformat()}")

    available_models = get_available_models()

    if not available_models:
        print("❌ No models available for code benchmarking")
        return 1

    print(f"\n✅ Available models: {len(available_models)}")
    for model_name in available_models:
        print(f"  - {model_name}")

    # Запускаем benchmarks
    benchmarks = ["humaneval+", "mbpp+"]
    launched = []

    for model_name, config in available_models.items():
        for benchmark in benchmarks:
            # Проверяем уже выполненные
            result_file = OUT / "codebench" / model_name / benchmark / "eval_results.json"
            if result_file.exists():
                print(f"⏭️  SKIP: {model_name}/{benchmark} (completed)")
                continue

            log_file = launch_code_benchmark(model_name, config, benchmark)
            launched.append((model_name, benchmark, log_file))

            # Небольшая пауза между запусками
            time.sleep(5)

    if launched:
        print(f"\n🎯 Launched {len(launched)} code benchmarks:")
        for model_name, benchmark, log_file in launched:
            print(f"  - {model_name}/{benchmark}: {log_file}")

        print(f"\n📊 Monitor progress:")
        print(f"  tail -f {LOGS}/codebench_*.log")
    else:
        print("\n⏭️  All benchmarks already completed")

    return 0


if __name__ == "__main__":
    exit(main())
