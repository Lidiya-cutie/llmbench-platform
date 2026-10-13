#!/usr/bin/env python3
"""
Оркестратор GPU ресурсов для множественных моделей.

Backward compatible: НЕ прерывает запущенные evaluate_model.py
Управляет запуском новых моделей на свободных GPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class GpuState:
    """Состояние GPU"""
    index: int
    memory_free_mb: int
    memory_total_mb: int
    utilization: float
    power_draw_w: float
    processes: List[subprocess.Popen]


@dataclass
class ModelTask:
    """Задача оценки модели"""
    model_key: str
    gpu_index: int
    suite: str = "simplified"
    base_url: Optional[str] = None
    served_model: Optional[str] = None
    wait_for: Optional[str] = None
    priority: int = 0
    status: str = "pending"  # pending, running, completed, failed


class GpuOrchestrator:
    """Оркестратор GPU ресурсов"""

    def __init__(self, config_path: str = "config/models.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.tasks: List[ModelTask] = []
        self.running_processes: Dict[str, subprocess.Popen] = {}

    def _load_config(self) -> dict:
        """Загрузка конфигурации моделей"""
        try:
            import yaml
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return {}

    def get_gpu_states(self) -> List[GpuState]:
        """Получение состояния GPU"""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,memory.free,memory.total,utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )

            states = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    states.append(GpuState(
                        index=int(parts[0]),
                        memory_free_mb=int(parts[1]),
                        memory_total_mb=int(parts[2]),
                        utilization=float(parts[3]),
                        power_draw_w=float(parts[4]),
                        processes=[]
                    ))
            return states
        except Exception as e:
            print(f"Ошибка получения состояния GPU: {e}")
            return []

    def add_task(self, task: ModelTask):
        """Добавление задачи в очередь"""
        self.tasks.append(task)
        print(f"Добавлена задача: {task.model_key} на GPU{task.gpu_index}")

    def can_run_on_gpu(self, gpu_index: int, model_key: str) -> bool:
        """Проверка возможности запуска на GPU"""
        gpu_states = self.get_gpu_states()

        for gpu in gpu_states:
            if gpu.index == gpu_index:
                # GPU должен быть свободен > 80%
                memory_free_ratio = gpu.memory_free_mb / gpu.memory_total_mb
                return memory_free_ratio > 0.8 and gpu.utilization < 20

        return False

    def start_task(self, task: ModelTask) -> Optional[subprocess.Popen]:
        """Запуск задачи оценки"""
        if not self.can_run_on_gpu(task.gpu_index, task.model_key):
            print(f"GPU{task.gpu_index} занят, откладывая {task.model_key}")
            return None

        cmd = self._build_command(task)
        log_path = f"logs/orchestrator_{task.model_key}_{datetime.now():%Y%m%d_%H%M%S}.log"

        try:
            with open(log_path, "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd="/opt/llmbench"
                )
                self.running_processes[task.model_key] = process
                task.status = "running"
                print(f"Запущен процесс {task.model_key} (PID: {process.pid})")
                return process
        except Exception as e:
            print(f"Ошибка запуска {task.model_key}: {e}")
            task.status = "failed"
            return None

    def _build_command(self, task: ModelTask) -> List[str]:
        """Построение команды запуска"""
        model_cfg = self.config["models"].get(task.model_key, {})
        defaults = self.config.get("defaults", {})

        base_url = task.base_url or model_cfg.get("base_url") or defaults.get("base_url")
        served_model = task.served_model or model_cfg.get("served_model_name", task.model_key)

        cmd = [
            "python3", "scripts/evaluate_model.py",
            "--model-key", task.model_key,
            "--base-url", base_url,
            "--served-model", served_model,
            "--suite", task.suite,
            "--out-dir", "out"
        ]

        return cmd

    def check_completed(self) -> List[str]:
        """Проверка завершённых процессов"""
        completed = []

        for model_key, process in list(self.running_processes.items()):
            if process.poll() is not None:
                completed.append(model_key)
                del self.running_processes[model_key]

                # Обновление статуса задачи
                for task in self.tasks:
                    if task.model_key == model_key:
                        task.status = "completed"

        return completed

    def run_orchestration(self, max_parallel: int = 2):
        """Запуск оркестрации"""
        print("Запуск оркестрации GPU ресурсов")

        while True:
            # Проверка завершённых
            completed = self.check_completed()
            if completed:
                print(f"Завершены: {', '.join(completed)}")

            # Проверка условий ожидания
            ready_tasks = []
            for task in self.tasks:
                if task.status == "pending":
                    # Проверка wait_for условия
                    if task.wait_for:
                        wait_task = next((t for t in self.tasks if t.model_key == task.wait_for), None)
                        if wait_task and wait_task.status == "completed":
                            ready_tasks.append(task)
                    else:
                        ready_tasks.append(task)

            # Запуск готовых задач с учётом параллельности
            running_count = len(self.running_processes)
            for task in ready_tasks:
                if running_count >= max_parallel:
                    break

                process = self.start_task(task)
                if process:
                    running_count += 1

            # Проверка завершения всех задач
            if all(t.status in ["completed", "failed"] for t in self.tasks):
                print("Все задачи завершены")
                break

            # Ожидание перед следующей проверкой
            time.sleep(60)

    def print_status(self):
        """Вывод статуса оркестрации"""
        gpu_states = self.get_gpu_states()

        print("\n=== Статус GPU ===")
        for gpu in gpu_states:
            free_pct = (gpu.memory_free_mb / gpu.memory_total_mb) * 100
            print(f"GPU{gpu.index}: {free_pct:.1f}% свободно ({gpu.memory_free_mb}/{gpu.memory_total_mb} MiB), "
                  f"{gpu.utilization}% загрузка")

        print("\n=== Задачи ===")
        for task in self.tasks:
            print(f"{task.model_key}: {task.status} (GPU{task.gpu_index})")

        print(f"\nЗапущено процессов: {len(self.running_processes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оркестратор GPU ресурсов")
    parser.add_argument("--config", default="config/models.yaml", help="Путь к конфигурации")
    parser.add_argument("--model", type=str, help="Модель для запуска")
    parser.add_argument("--gpu", type=int, help="GPU для модели")
    parser.add_argument("--suite", default="simplified", help="Suite для оценки")
    parser.add_argument("--wait-for", type=str, help="Ожидание завершения модели")
    parser.add_argument("--status", action="store_true", help="Статус оркестрации")
    parser.add_argument("--max-parallel", type=int, default=2, help="Максимальная параллельность")

    args = parser.parse_args()

    orchestrator = GpuOrchestrator(args.config)

    if args.status:
        orchestrator.print_status()

    elif args.model and args.gpu is not None:
        task = ModelTask(
            model_key=args.model,
            gpu_index=args.gpu,
            suite=args.suite,
            wait_for=args.wait_for
        )
        orchestrator.add_task(task)
        orchestrator.run_orchestration(args.max_parallel)

    else:
        parser.print_help()
