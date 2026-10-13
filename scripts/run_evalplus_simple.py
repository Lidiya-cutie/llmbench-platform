#!/usr/bin/env python3
"""Простой запуск EvalPlus без лишних параметров.

Только базовая функциональность для тестирования.
"""
import subprocess
import sys
from pathlib import Path

def run_evalplus_simple(model_key: str, benchmark: str = "humaneval"):
    """Простой запуск EvalPlus."""

    # Базовые конфигурации
    # local-27b использует llama-server с OpenAI-совместимым API -> backend=openai
    model_configs = {
        "local-27b-27b-ternary": {
            "backend": "openai",  # llama-server поддерживает OpenAI API
            "model": "Ternary-local-27b-Q2_0.gguf",
            "base_url": "http://127.0.0.1:8082/v1",
        },
        "gemma-3-27b-awq": {
            "backend": "openai",  # Triton с OpenAI-совместимым API
            "model": "gemma-3-27b-awq",
            "base_url": "http://127.0.0.1:19892/v1",
        },
        "ollama-qwen3-8b": {
            "backend": "openai",  # Ollama с OpenAI-совместимым API
            "model": "qwen3:8b",
            "base_url": "http://127.0.0.1:11434/v1",
        },
        "deepseek-v4-flash": {
            "backend": "openai",
            "model": "deepseek-v4-flash-dspark",
            "base_url": "http://31.28.10.135:8000/v1",
        },
    }

    if model_key not in model_configs:
        print(f"ERROR: Unknown model {model_key}")
        return 1

    config = model_configs[model_key]

    # Минимальная команда EvalPlus
    cmd = [
        sys.executable,
        "-m", "evalplus.evaluate",
        benchmark,  # humaneval или mbpp
        "--backend", config["backend"],
        "--model", config["model"],
        "--base-url", config["base_url"],
    ]

    print(f"Running: {' '.join(cmd)}")

    # Запуск
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
    print("\nSTDERR:")
    print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
    print(f"\nReturn code: {result.returncode}")

    return result.returncode

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_evalplus_simple.py <model_key> [benchmark]")
        sys.exit(1)

    model = sys.argv[1]
    benchmark = sys.argv[2] if len(sys.argv) > 2 else "humaneval"

    exit(run_evalplus_simple(model, benchmark))
