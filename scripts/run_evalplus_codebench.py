#!/usr/bin/env python3
"""EvalPlus code benchmark integration для /opt/llm-data/llmbench.

Поддержка HumanEval+, MBPP+, LiveCodeBench через единый harness.
Интеграция с существующей системой карточек и дашбордов.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Конфигурация бенчмарков
CODE_BENCHMARKS = {
    "humaneval+": {
        "dataset": "humaneval",
        "plus": True,
        "expected_range": (0.75, 0.95),  # 75-95% pass@1 для современных моделей
        "version": "0.1.10",
    },
    "mbpp+": {
        "dataset": "mbpp",
        "plus": True,
        "expected_range": (0.70, 0.90),
        "version": "0.2.1",
    },
}

# Backend конфигурации для разных типов моделей
BACKEND_CONFIGS = {
    "vllm": {
        "command": "evalplus evaluate",
        "base_args": ["--backend", "vllm", "--greedy"],
        "reasoning_support": True,
    },
    "ollama": {
        "command": "evalplus evaluate",
        "base_args": ["--backend", "ollama", "--greedy"],
        "reasoning_support": True,
    },
    "openai": {
        "command": "evalplus evaluate",
        "base_args": ["--backend", "openai", "--greedy"],
        "reasoning_support": False,
    }
}

# Модель-специфичные конфигурации
MODEL_CONFIGS = {
    "qwen3-8b-bf16": {
        "backend": "vllm",
        "base_url": "http://127.0.0.1:8077/v1",
        "model": "Qwen3-8B",
        "max_tokens": 4096,  # Для reasoning
        "reasoning_parser": "qwen3",
    },
    "qwen2.5-vl-7b": {
        "backend": "vllm",
        "base_url": "http://127.0.0.1:8077/v1",
        "model": "Qwen2.5-VL-7B-Instruct",
        "max_tokens": 2048,
        "reasoning_parser": None,
    },
    "ollama-qwen3-8b": {
        "backend": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen3:8b",
        "max_tokens": 4096,
        "reasoning_parser": "qwen3",
    },
    "local-27b-27b-ternary": {
        "backend": "ollama",  # llama-server с OpenAI-совместимым API
        "base_url": "http://127.0.0.1:8082/v1",
        "model": "Ternary-local-27b-Q2_0.gguf",
        "max_tokens": 2048,
        "reasoning_parser": None,
    },
    "gemma-3-27b-awq": {
        "backend": "vllm",
        "base_url": "http://127.0.0.1:19892/v1",
        "model": "gemma-3-27b-awq",
        "max_tokens": 2048,
        "reasoning_parser": "deepseek_r1",  # Gemma thinking mode
    },
    "deepseek-v4-flash": {
        "backend": "openai",
        "base_url": "http://31.28.10.135:8000/v1",
        "model": "deepseek-v4-flash-dspark",
        "max_tokens": 4096,
        "reasoning_parser": "deepseek_r1",
    },
}


class CodeBenchmarkRunner:
    """Запуск code benchmarks с интеграцией в /opt/llm-data/llmbench."""

    def __init__(self, llmbench_dir: Path):
        self.llmbench = llmbench_dir
        self.out_dir = llmbench_dir / "out"
        self.code_dir = self.out_dir / "codebench"
        self.logs_dir = llmbench_dir / "logs"

    def run_benchmark(
        self,
        model_key: str,
        benchmark: str = "humaneval+",
        checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """Запуск одного code benchmark."""

        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")

        if benchmark not in CODE_BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        model_config = MODEL_CONFIGS[model_key]
        bench_config = CODE_BENCHMARKS[benchmark]
        backend_config = BACKEND_CONFIGS[model_config["backend"]]

        # Подготовка директорий
        model_dir = self.code_dir / model_key
        benchmark_dir = model_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Лог файл
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"codebench_{model_key}_{benchmark}_{timestamp}.log"

        # Формирование команды EvalPlus через Python module
        cmd = [
            sys.executable,  # Текущий Python интерпретатор
            "-m", "evalplus.evaluate",
            bench_config['dataset'],  # Dataset как позиционный аргумент
        ]

        # Backend-specific аргументы
        if model_config["backend"] == "vllm":
            cmd.extend([
                "--backend", "vllm",
                "--model", model_config['model'],
                "--base-url", model_config['base_url'],
            ])

        elif model_config["backend"] == "ollama":
            cmd.extend([
                "--backend", "ollama",
                "--model", model_config['model'],
                "--base-url", model_config['base_url'],
            ])

        elif model_config["backend"] == "openai":
            cmd.extend([
                "--backend", "openai",
                "--model", model_config['model'],
                "--base-url", model_config['base_url'],
            ])

        # Общие флаги
        cmd.append("--greedy")  # pass@1

        # EvalPlus специфичные флаги - для HumanEval+ нужно использовать dataset name без +
        # Используем базовый dataset, EvalPlus сам определит нужен ли +
        if bench_config["plus"]:
            # Для HumanEval+ и MBPP+ нужно использовать обычное имя dataset
            pass  # EvalPlus сам обработает плюс версию если нужно

        if checkpoint:
            cmd.extend([
                "--checkpoint",
                "--checkpoint-dir", str(benchmark_dir / ".evalplus_cache"),
            ])

        # Input директория
        cmd.extend([
            "--i",
            str(self.llmbench / ".cache" / "evalplus" / bench_config['dataset']),
        ])

        # Запуск с логированием
        print(f"Running: {' '.join(cmd)}")
        print(f"Log: {log_file}")

        try:
            with open(log_file, "w") as log:
                result = subprocess.run(
                    cmd,
                    cwd=self.llmbench,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 час
                )

                log.write(result.stdout)
                log.write(result.stderr)

                if result.returncode != 0:
                    raise RuntimeError(f"EvalPlus failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"EvalPlus timeout for {model_key}/{benchmark}")
        except Exception as e:
            raise RuntimeError(f"EvalPlus error: {e}")

        # Парсинг результатов
        return self._parse_results(result.stdout, benchmark_dir)

    def _parse_results(self, output: str, result_dir: Path) -> Dict[str, Any]:
        """Парсинг результатов EvalPlus."""

        # EvalPlus выводит JSON results
        results_file = result_dir / "eval_results.json"

        if results_file.exists():
            results = json.loads(results_file.read_text())
            return results

        # Fallback: парсинг из stdout
        return {"raw_output": output, "status": "parsed_from_stdout"}

    def update_card(self, model_key: str, benchmark_results: Dict[str, Any]):
        """Обновление карточки модели с code benchmark results."""

        card_file = self.out_dir / f"card_{model_key}.json"

        if not card_file.exists():
            raise FileNotFoundError(f"Card not found: {card_file}")

        card = json.loads(card_file.read_text())

        # Добавляем/code секцию если нет
        if "code" not in card:
            card["code"] = {}

        # Обновляем результаты
        for bench_name, results in benchmark_results.items():
            if "pass@1" in results:
                card["code"][bench_name] = {
                    "pass@1": results["pass@1"],
                    "pass@10": results.get("pass@10", None),
                    "samples": results.get("n_samples", 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        # Backup
        backup_file = card_file.with_suffix(f".json.bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.copy(card_file, backup_file)

        # Write updated card
        card_file.write_text(json.dumps(card, indent=2, ensure_ascii=False))

        print(f"Updated card: {card_file}")
        print(f"Backup: {backup_file}")


def main():
    parser = argparse.ArgumentParser(description="EvalPlus integration for llmbench")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--benchmark", default="humaneval+", choices=list(CODE_BENCHMARKS.keys()))
    parser.add_argument("--llmbench", default="/opt/llm-data/llmbench", type=Path)
    parser.add_argument("--update-card", action="store_true")

    args = parser.parse_args()

    runner = CodeBenchmarkRunner(args.llmbench)

    try:
        results = runner.run_benchmark(args.model, args.benchmark)
        print(f"Results: {json.dumps(results, indent=2)}")

        if args.update_card:
            runner.update_card(args.model, {args.benchmark: results})

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
