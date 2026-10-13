#!/usr/bin/env python3
"""Надежная система очередей с автоматическим перезапуском при падениях.

Проблемы предыдущей системы:
1. EvalPlus падал при CLI вызове (не в PATH)
2. LM-EVAL падал без перезапуска
3. Нет механизма пропуска недоступных моделей
4. Нет автоматического повторного запуска

Решение:
- Автоматический перезапуск при падениях (до 3 попыток)
- Checkpoint система для отслеживания прогресса
- Пропуски недоступных моделей с логированием
- Механизм продолжения очереди при одиночных падениях
"""
from __future__ import annotations

import json
import subprocess
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

LLMBENCH = Path("/opt/llm-data/llmbench")
LOGS_DIR = LLMBENCH / "logs"
OUT_DIR = LLMBENCH / "out"

# Максимальное количество попыток перезапуска
MAX_RETRIES = 3
# Задержка между попытками
RETRY_DELAY = 60  # секунды


class ReliableQueue:
    """Надежная система очередей с обработкой ошибок."""

    def __init__(self):
        self.checkpoint_file = LOGS_DIR / "reliable_queue.checkpoint"
        self.error_log = LOGS_DIR / "reliable_queue_errors.log"
        self.checkpoint = self._load_checkpoint()
        self.errors = []

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Загрузка checkpoint файла."""
        if self.checkpoint_file.exists():
            try:
                return json.loads(self.checkpoint_file.read_text())
            except Exception:
                return {}
        return {"completed": [], "failed": [], "retries": {}}

    def _save_checkpoint(self):
        """Сохранение checkpoint файла."""
        self.checkpoint_file.write_text(json.dumps(self.checkpoint, indent=2))

    def _log_error(self, task_id: str, error: str):
        """Логирование ошибки."""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = f"[{timestamp}] {task_id}: {error}\n"

        with open(self.error_log, "a") as f:
            f.write(log_entry)

        self.errors.append({
            "task": task_id,
            "error": error,
            "timestamp": timestamp
        })

    def _check_api_ready(self, model_config: Dict[str, Any]) -> bool:
        """Проверка доступности API модели."""
        try:
            base_url = model_config["base_url"]
            endpoints = [
                f"{base_url}/models",
                f"{base_url}/v1/models",
                f"{base_url.replace('/v1', '')}/api/tags"  # Ollama
            ]

            for endpoint in endpoints:
                try:
                    resp = requests.get(endpoint, timeout=5)
                    if resp.status_code == 200:
                        return True
                except Exception:
                    continue

            return False
        except Exception:
            return False

    def _run_task_safe(
        self,
        task_id: str,
        command: List[str],
        log_file: Path,
        retry_count: int = 0
    ) -> bool:
        """Безопасный запуск задачи с перезапуском при падениях."""

        if task_id in self.checkpoint["completed"]:
            print(f"⏭️  SKIP: {task_id} (уже завершено)")
            return True

        # Проверка количества попыток
        retry_key = f"{task_id}_retry_{retry_count}"
        if retry_key in self.checkpoint["retries"]:
            if self.checkpoint["retries"][retry_key] >= MAX_RETRIES:
                print(f"❌ SKIP: {task_id} (достигнут лимит попыток)")
                self._log_error(task_id, f"Достигнут лимит попыток ({MAX_RETRIES})")
                return False

        print(f"🚀 Запуск: {task_id} (попытка {retry_count + 1}/{MAX_RETRIES})")

        try:
            with open(log_file, "w") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=LLMBENCH,
                    timeout=7200,  # 2 часа
                )

            success = result.returncode == 0

            if success:
                print(f"✅ SUCCESS: {task_id}")
                if task_id not in self.checkpoint["completed"]:
                    self.checkpoint["completed"].append(task_id)
                self._save_checkpoint()
                return True
            else:
                raise RuntimeError(f"Exit code: {result.returncode}")

        except subprocess.TimeoutExpired:
            error = f"Timeout (2 часа)"
            print(f"⏱️  TIMEOUT: {task_id}")
            self._log_error(task_id, error)
        except Exception as e:
            error = str(e)
            print(f"❌ ERROR: {task_id} - {error}")
            self._log_error(task_id, error)

        # Попытка перезапуска
        if retry_count < MAX_RETRIES - 1:
            print(f"🔄 Перезапуск через {RETRY_DELAY} сек...")
            time.sleep(RETRY_DELAY)

            # Обновляем счетчик попыток
            self.checkpoint["retries"][retry_key] = retry_count + 1
            self._save_checkpoint()

            return self._run_task_safe(task_id, command, log_file, retry_count + 1)
        else:
            print(f"❌ FINAL FAIL: {task_id} (все попытки исчерпаны)")
            if task_id not in self.checkpoint["failed"]:
                self.checkpoint["failed"].append(task_id)
            self._save_checkpoint()
            return False

    def run_lm_eval_task(
        self,
        model_key: str,
        base_url: str,
        model_name: str,
        task: str,
        checkpoint: bool = True
    ) -> bool:
        """Запуск lm-eval задачи с надежностью."""

        task_id = f"lm_eval_{model_key}_{task}"
        log_file = LOGS_DIR / f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Проверка доступности API
        model_config = {"base_url": base_url}
        if not self._check_api_ready(model_config):
            print(f"⚠️  SKIP: {task_id} (API недоступен)")
            self._log_error(task_id, "API недоступен")
            return False

        command = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-chat-completions",
            "--model_args", f"base_url={base_url}/chat/completions,model={model_name},num_concurrent=1,tokenized_requests=False",
            "--tasks", task,
            "--apply_chat_template",
            "--output_path", str(OUT_DIR / f"lmeval_{model_key}_queue" / task),
            "--log_samples",
            "--use_cache", str(LLMBENCH / ".cache" / f"lm_eval_{model_key}_{task}"),
            "--cache_requests", "true",
            "--confirm_run_unsafe_code",
        ]

        return self._run_task_safe(task_id, command, log_file)

    def run_code_benchmark(
        self,
        model_key: str,
        base_url: str,
        model_name: str,
        benchmark: str = "humaneval",
        backend: str = "openai"
    ) -> bool:
        """Запуск code benchmark с надежностью."""

        task_id = f"codebench_{model_key}_{benchmark}"
        log_file = LOGS_DIR / f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Проверка доступности API
        model_config = {"base_url": base_url}
        if not self._check_api_ready(model_config):
            print(f"⚠️  SKIP: {task_id} (API недоступен)")
            self._log_error(task_id, "API недоступен")
            return False

        command = [
            sys.executable, "-m", "evalplus.evaluate",
            benchmark,  # humaneval или mbpp
            "--backend", backend,
            "--model", model_name,
            "--base-url", base_url,
            "--greedy",
        ]

        return self._run_task_safe(task_id, command, log_file)

    def print_summary(self):
        """Вывод summary отчета."""
        print("\n" + "="*60)
        print("📊 СТАТИСТИКА ВЫПОЛНЕНИЯ ОЧЕРЕДИ")
        print("="*60)

        completed = len(self.checkpoint["completed"])
        failed = len(self.checkpoint["failed"])
        total = completed + failed

        print(f"✅ Завершено успешно: {completed}")
        print(f"❌ Завершено с ошибками: {failed}")
        print(f"📊 Всего попыток: {total}")

        if self.errors:
            print(f"\n❌ ОШИБКИ ({len(self.errors)}):")
            for error in self.errors[-5:]:  # Последние 5 ошибок
                print(f"   • {error['task']}: {error['error']}")

        print("="*60)


def main():
    """Главная функция с полным набором задач."""

    queue = ReliableQueue()

    print("🔄 НАДЕЖНАЯ СИСТЕМА ОЧЕРЕДЕЙ С АВТОПЕРЕЗАПУСКОМ")
    print("="*60)

    # Конфигурация моделей
    models = {
        "local-27b-27b-ternary": {
            "base_url": "http://127.0.0.1:8082/v1",
            "model_name": "Ternary-local-27b-Q2_0.gguf",
            "tasks": ["gsm8k", "minerva_math", "mmlu_pro", "humaneval"],
            "code_benchmarks": ["humaneval", "mbpp"],
        },
        "ollama-qwen3-8b": {
            "base_url": "http://127.0.0.1:11434/v1",
            "model_name": "qwen3:8b",
            "tasks": ["gsm8k", "minerva_math", "mmlu_pro"],
            "code_benchmarks": ["humaneval", "mbpp"],
        },
        "gemma-3-27b-awq": {
            "base_url": "http://127.0.0.1:19892/v1",
            "model_name": "gemma-3-27b-awq",
            "tasks": ["mmlu_pro"],
            "code_benchmarks": ["humaneval", "mbpp"],
        },
    }

    total_tasks = sum(len(m["tasks"]) + len(m["code_benchmarks"]) for m in models.values())
    print(f"📋 Всего задач: {total_tasks}")
    print()

    # Запуск всех задач
    completed_count = 0

    for model_key, config in models.items():
        print(f"\n🔧 Модель: {model_key}")

        # LM-EVAL задачи
        for task in config["tasks"]:
            if queue.run_lm_eval_task(
                model_key=model_key,
                base_url=config["base_url"],
                model_name=config["model_name"],
                task=task
            ):
                completed_count += 1

        # Code benchmarks
        for benchmark in config["code_benchmarks"]:
            if queue.run_code_benchmark(
                model_key=model_key,
                base_url=config["base_url"],
                model_name=config["model_name"],
                benchmark=benchmark,
            ):
                completed_count += 1

    # Итоговый отчет
    queue.print_summary()

    print(f"\n🎯 РЕЗУЛЬТАТ: {completed_count}/{total_tasks} задач выполнено успешно")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем")
        exit(1)
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        exit(1)
