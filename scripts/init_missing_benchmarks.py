#!/usr/bin/env python3
"""
Инициализация недостающих бенчмарков с приоритетами
"""

from benchmark_queue_manager import BenchmarkQueue

def init_missing_benchmarks():
    """Создание очереди для всех недостающих бенчмарков"""

    queue = BenchmarkQueue()

    # Приоритеты:
    # 10 - критические (блокируют другие задачи)
    # 8-9 - высокие
    # 5-7 - средние
    # 1-4 - низкие

    benchmarks = [
        # КРИТИЧЕСКИЕ - запущены, но нужны дополнительные метрики
        {"model": "deepseek-v4-flash", "task": "mmlu_pro", "priority": 10, "gpu": None},

        # ВЫСОКИЙ ПРИОРИТЕТ - модели с хорошей производительностью
        {"model": "gemma-3-27b-awq", "task": "mbpp", "priority": 9, "gpu": 0},
        {"model": "gemma-3-27b-awq", "task": "minerva_math", "priority": 9, "gpu": 0},
        {"model": "qwen3-8b-bf16", "task": "mmlu_pro", "priority": 8, "gpu": 1},

        # СРЕДНИЙ ПРИОРИТЕТ - модели средней производительности
        {"model": "glm-4-9b-chat", "task": "humaneval", "priority": 7, "gpu": 1},
        {"model": "qwen2.5-vl-7b", "task": "mmlu_pro", "priority": 7, "gpu": 1},

        # НИЗКИЙ ПРИОРИТЕТ - менее важные бенчмарки
        {"model": "glm-4-9b-chat", "task": "mbpp", "priority": 5, "gpu": 1},
        {"model": "glm-4-9b-chat", "task": "gpqa", "priority": 4, "gpu": 1},
        {"model": "ollama-qwen3-8b", "task": "humaneval", "priority": 5, "gpu": None},
        {"model": "ollama-qwen3-8b", "task": "minerva_math", "priority": 4, "gpu": None},
        {"model": "ollama-qwen3-8b", "task": "gpqa", "priority": 3, "gpu": None},
        {"model": "qwen2.5-vl-7b", "task": "humaneval", "priority": 5, "gpu": 1},
        {"model": "qwen2.5-vl-7b", "task": "mbpp", "priority": 4, "gpu": 1},
        {"model": "qwen2.5-vl-7b", "task": "gpqa", "priority": 3, "gpu": 1},

        # ОЧЕНЬ НИЗКИЙ - local-27b-1bit (требует полного пересчёта)
        {"model": "local-27b-1bit", "task": "humaneval", "priority": 2, "gpu": None},
        {"model": "local-27b-1bit", "task": "mbpp", "priority": 2, "gpu": None},
        {"model": "local-27b-1bit", "task": "mmlu_pro", "priority": 2, "gpu": None},
        {"model": "local-27b-1bit", "task": "gsm8k", "priority": 2, "gpu": None},
        {"model": "local-27b-1bit", "task": "minerva_math", "priority": 1, "gpu": None},
        {"model": "local-27b-1bit", "task": "gpqa", "priority": 1, "gpu": None},
    ]

    # Добавляем задачи в очередь
    added_jobs = []
    for job_spec in benchmarks:
        job_id = queue.add_benchmark(
            model=job_spec["model"],
            task=job_spec["task"],
            priority=job_spec["priority"],
            gpu_id=job_spec.get("gpu")
        )
        added_jobs.append(job_id)

    print(f"Добавлено {len(added_jobs)} задач в очередь")
    queue.show_status()

    return added_jobs

if __name__ == "__main__":
    init_missing_benchmarks()
