#!/usr/bin/env python3
"""
Expert Benchmark Queue Manager for Production
Горизонтально масштабируемая система очередей с приоритетами
"""

import json
import subprocess
import time
import os
from datetime import datetime
from pathlib import Path
import argparse

class BenchmarkQueue:
    def __init__(self, base_dir="/opt/llmbench"):
        self.base_dir = Path(base_dir)
        self.out_dir = self.base_dir / "out"
        self.logs_dir = self.base_dir / "logs"
        self.queue_file = self.logs_dir / "benchmark_queue.json"
        self.status_file = self.logs_dir / "queue_status.json"

        # Создаём директории
        self.logs_dir.mkdir(exist_ok=True)

        # Загружаем или создаём очередь
        self.queue = self.load_queue()
        self.status = self.load_status()

    def load_queue(self):
        if self.queue_file.exists():
            with open(self.queue_file) as f:
                return json.load(f)
        return {"pending": [], "running": [], "completed": [], "failed": []}

    def save_queue(self):
        with open(self.queue_file, 'w') as f:
            json.dump(self.queue, f, indent=2)

    def load_status(self):
        if self.status_file.exists():
            with open(self.status_file) as f:
                return json.load(f)
        return {"last_update": None, "gpu_status": {}}

    def save_status(self):
        self.status["last_update"] = datetime.utcnow().isoformat()
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def check_gpu_available(self, gpu_id=None):
        """Проверка доступности GPU с учетом текущих процессов"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )

            for line in result.stdout.strip().split('\n'):
                idx, util, mem_used, mem_total = line.split(', ')
                idx = int(idx)
                util = int(util)
                mem_used = int(mem_used)
                mem_total = int(mem_total)

                # GPU доступен если утилизация < 80% и памяти свободно > 8GB
                available = util < 80 and (mem_total - mem_used) > 8000

                if gpu_id is None or gpu_id == idx:
                    self.status["gpu_status"][f"GPU{idx}"] = {
                        "utilization": util,
                        "memory_used_mb": mem_used,
                        "memory_total_mb": mem_total,
                        "memory_free_mb": mem_total - mem_used,
                        "available": available
                    }

            return self.status["gpu_status"]
        except Exception as e:
            print(f"Ошибка проверки GPU: {e}")
            return {}

    def add_benchmark(self, model, task, priority=5, gpu_id=None, depends_on=None):
        """Добавление бенчмарка в очередь"""
        job = {
            "id": f"{model}_{task}_{int(time.time())}",
            "model": model,
            "task": task,
            "priority": priority,  # 1-10, где 10 - наивысший
            "gpu_id": gpu_id,
            "depends_on": depends_on or [],
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "result": None
        }

        self.queue["pending"].append(job)
        self.save_queue()
        return job["id"]

    def get_next_job(self):
        """Получение следующей задачи с учётом приоритетов и зависимостей"""
        if not self.queue["pending"]:
            return None

        # Сортируем по приоритету (высокий первый)
        sorted_jobs = sorted(self.queue["pending"], key=lambda x: -x["priority"])

        for job in sorted_jobs:
            # Проверяем зависимости
            deps_satisfied = all(
                dep_id in [j["id"] for j in self.queue["completed"]]
                for dep_id in job["depends_on"]
            )

            if deps_satisfied:
                return job

        return None

    def start_job(self, job):
        """Запуск задачи"""
        job["status"] = "running"
        job["started_at"] = datetime.utcnow().isoformat()

        self.queue["pending"].remove(job)
        self.queue["running"].append(job)
        self.save_queue()

        return job

    def complete_job(self, job_id, success=True, result=None):
        """Завершение задачи"""
        job = None
        for j in self.queue["running"]:
            if j["id"] == job_id:
                job = j
                break

        if job:
            job["status"] = "completed" if success else "failed"
            job["completed_at"] = datetime.utcnow().isoformat()
            job["result"] = result

            self.queue["running"].remove(job)
            if success:
                self.queue["completed"].append(job)
            else:
                self.queue["failed"].append(job)

            self.save_queue()

    def build_command(self, job):
        """Построение команды для запуска бенчмарка"""
        model = job["model"]
        task = job["task"]
        gpu_id = job.get("gpu_id")

        # Карта конфигураций моделей
        model_configs = {
            "gemma-3-27b-awq": {
                "base_url": "http://127.0.0.1:19892/v1/chat/completions",
                "model": "gemma-3-27b-awq",
                "num_concurrent": 4
            },
            "qwen3-8b-bf16": {
                "base_url": "http://127.0.0.1:8082/v1/chat/completions",
                "model": "Qwen3-8B",
                "num_concurrent": 4
            },
            "glm-4-9b-chat": {
                "base_url": "http://127.0.0.1:8082/v1/chat/completions",
                "model": "glm-4-9b-chat",
                "num_concurrent": 4
            },
            "ollama-qwen3-8b": {
                "base_url": "http://127.0.0.1:11434/v1/chat/completions",
                "model": "qwen3:8b",
                "num_concurrent": 2
            },
            "qwen2.5-vl-7b": {
                "base_url": "http://127.0.0.1:8082/v1/chat/completions",
                "model": "qwen2.5-vl-7b",
                "num_concurrent": 4
            },
            "local-27b-27b-ternary": {
                "base_url": "http://127.0.0.1:8681/v1/chat/completions",
                "model": "local-27b-27b-ternary",
                "num_concurrent": 2
            },
            "deepseek-v4-flash": {
                "base_url": "http://31.28.10.135:8000/v1/chat/completions",
                "model": "deepseek-v4-flash-dspark",
                "num_concurrent": 2
            }
        }

        config = model_configs.get(model)
        if not config:
            return None

        output_path = f"out/lmeval_{model}_queue/{task}/{model}"

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "local-chat-completions",
            "--model_args", f"model={config['model']},base_url={config['base_url']},num_concurrent={config['num_concurrent']},max_length=4096,max_gen_toks=2048,tokenized_requests=False",
            "--tasks", task,
            "--apply_chat_template",
            "--batch_size", "1",
            "--output_path", output_path,
            "--log_samples"
        ]

        return cmd

    def run_job(self, job):
        """Запуск задачи с монитором"""
        cmd = self.build_command(job)
        if not cmd:
            print(f"Не удалось построить команду для {job['model']} {job['task']}")
            return False

        print(f"Запуск задачи: {job['id']}")
        print(f"Команда: {' '.join(cmd)}")

        log_file = self.logs_dir / f"job_{job['id']}.log"

        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=self.base_dir
                )

                # Мониторинг процесса
                while True:
                    ret = process.poll()
                    if ret is not None:
                        success = ret == 0
                        break

                    time.sleep(10)  # Проверяем каждые 10 секунд

            result = {"log_file": str(log_file)}
            self.complete_job(job["id"], success, result)

            return success

        except Exception as e:
            print(f"Ошибка выполнения задачи {job['id']}: {e}")
            self.complete_job(job["id"], False, {"error": str(e)})
            return False

    def show_status(self):
        """Отображение статуса очереди"""
        print("=== СТАТУС ОЧЕРЕДИ БЕНЧМАРКОВ ===")
        print(f"Последнее обновление: {self.status.get('last_update', 'N/A')}")
        print()

        if self.status.get("gpu_status"):
            print("=== GPU СТАТУС ===")
            for gpu_id, status in self.status["gpu_status"].items():
                available = "✅ Доступен" if status["available"] else "❌ Занят"
                print(f"{gpu_id}: {available}")
                print(f"   Утилизация: {status['utilization']}%")
                print(f"   Память: {status['memory_used_mb']}/{status['memory_total_mb']} MB")
                print(f"   Свободно: {status['memory_free_mb']} MB")
                print()

        print(f"=== ОЧЕРЕДЬ ===")
        print(f"Ожидает: {len(self.queue['pending'])}")
        print(f"Выполняется: {len(self.queue['running'])}")
        print(f"Завершено: {len(self.queue['completed'])}")
        print(f"Ошибок: {len(self.queue['failed'])}")
        print()

        if self.queue["pending"]:
            print("=== ОЖИДАЮЩИЕ ЗАДАЧИ (топ-5) ===")
            for job in sorted(self.queue["pending"], key=lambda x: -x["priority"])[:5]:
                print(f"  [{job['priority']}] {job['model']} - {job['task']}")

        if self.queue["running"]:
            print("=== ВЫПОЛНЯЮЩИЕСЯ ЗАДАЧИ ===")
            for job in self.queue["running"]:
                elapsed = datetime.utcnow() - datetime.fromisoformat(job["started_at"])
                print(f"  {job['model']} - {job['task']} ({elapsed.seconds // 60} мин)")


def main():
    parser = argparse.ArgumentParser(description="Expert Benchmark Queue Manager")
    parser.add_argument("--action", choices=["add", "run", "status", "init"], default="status")
    parser.add_argument("--model", help="Модель для бенчмарка")
    parser.add_argument("--task", help="Задача бенчмарка")
    parser.add_argument("--priority", type=int, default=5, help="Приоритет (1-10)")
    parser.add_argument("--gpu", type=int, help="GPU ID")

    args = parser.parse_args()

    queue = BenchmarkQueue()

    if args.action == "status":
        queue.check_gpu_available()
        queue.show_status()

    elif args.action == "add":
        if not args.model or not args.task:
            print("Ошибка: --model и --task обязательны для добавления")
            return

        job_id = queue.add_benchmark(args.model, args.task, args.priority, args.gpu)
        print(f"Добавлена задача: {job_id}")

    elif args.action == "run":
        queue.check_gpu_available()

        while True:
            job = queue.get_next_job()
            if not job:
                print("Нет задач для выполнения")
                break

            # Проверяем GPU
            gpu_status = queue.status.get("gpu_status", {})
            available_gpus = [gpu_id for gpu_id, status in gpu_status.items() if status["available"]]

            if not available_gpus:
                print("Нет доступных GPU. Ожидание...")
                time.sleep(60)
                continue

            # Запускаем задачу
            job = queue.start_job(job)
            queue.run_job(job)

            # Обновляем статус
            queue.check_gpu_available()
            queue.save_status()

    elif args.action == "init":
        print("Инициализация экспертной системы очередей...")
        print("Созданы файлы:")
        print(f"  - {queue.queue_file}")
        print(f"  - {queue.status_file}")


if __name__ == "__main__":
    main()
