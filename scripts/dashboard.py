#!/usr/bin/env python3
"""
Простой dashboard для отслеживания прогресса моделей в реальном времени.

Отображает:
- Статус активных прогонов
- Прогресс по количеству запросов
- Оставшееся время
- Состояние GPU
"""

import json
import time
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List


class ProgressTracker:
    def __init__(self):
        self.logs_dir = Path("/opt/llmbench/logs")
        self.cards_dir = Path("/opt/llmbench/out")

    def get_active_models(self) -> List[Dict[str, Any]]:
        """Получение списка активных моделей."""
        active = []

        # Проверка процессов evaluate_model
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )

            for line in result.stdout.split('\n'):
                if 'evaluate_model.py' in line and '--model-key' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if '--model-key' in part and i + 1 < len(parts):
                            model_key = parts[i + 1]
                            pid = int(line.split()[1])

                            # Проверка лога для получения прогресса
                            progress = self.get_model_progress(model_key)

                            active.append({
                                "model": model_key,
                                "pid": pid,
                                "progress": progress,
                                "status": "running"
                            })
                            break
        except Exception as e:
            print(f"Error getting active models: {e}")

        return active

    def get_model_progress(self, model_key: str) -> Dict[str, Any]:
        """Получение прогресса из логов модели."""
        log_files = {
            "qwen3-8b-bf16": "eval_qwen3-8b-bf16_fixed.log",
            "qwen2.5-vl-7b": "eval_qwen2.5-vl-7b.log",
            "glm-4-9b-chat": "eval_glm-4-9b-chat.log",
            "local-27b-27b-ternary": "eval_local-27b_ternary_gpu1.log"
        }

        log_file = self.logs_dir / log_files.get(model_key, "")
        if not log_file.exists():
            return {"status": "no_log", "current": "N/A", "total": "N/A"}

        try:
            with open(log_file) as f:
                lines = f.readlines()

            # Поиск последней строки с прогрессом
            for line in reversed(lines[-50:]):
                if "Requesting API:" in line:
                    # Извлечение прогресса
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if '/' in part and 'Requesting API:' not in part:
                            try:
                                current, total = part.split('/')
                                return {
                                    "status": "active",
                                    "current": int(current),
                                    "total": int(total),
                                    "percent": round(int(current) / int(total) * 100, 2)
                                }
                            except (ValueError, IndexError):
                                continue
        except Exception:
            pass

        return {"status": "unknown", "current": "N/A", "total": "N/A"}

    def get_gpu_status(self) -> List[Dict[str, Any]]:
        """Получение статуса GPU."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader"],
                capture_output=True,
                text=True
            )

            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_used": parts[2],
                            "memory_total": parts[3],
                            "utilization": parts[4],
                            "temperature": parts[5]
                        })
            return gpus
        except Exception as e:
            return [{"error": str(e)}]

    def get_cards_status(self) -> Dict[str, Any]:
        """Получение статуса карточек."""
        cards = {}
        for card_file in self.cards_dir.glob("card_*.json"):
            try:
                with open(card_file) as f:
                    card = json.load(f)
                    model_name = card.get("name", "unknown")
                    quality = card.get("quality", {})
                    generated_at = card.get("generated_at", "")

                    cards[model_name] = {
                        "file": card_file.name,
                        "quality_empty": len(quality) == 0,
                        "generated_at": generated_at,
                        "updated_recently": self.is_recent(generated_at)
                    }
            except Exception:
                continue

        return cards

    def is_recent(self, timestamp_str: str, hours: int = 6) -> bool:
        """Проверка актуальности времени."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return datetime.now() - dt < timedelta(hours=hours)
        except:
            return False

    def print_dashboard(self):
        """Вывод dashboard в консоль."""
        print("\n" + "=" * 80)
        print(f" LLMBENCH DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Активные модели
        print("\n🔄 АКТИВНЫЕ ПРОГОНЫ:")
        active = self.get_active_models()
        if not active:
            print("   Нет активных прогонов")
        else:
            for model in active:
                if model['progress']['status'] == 'active':
                    print(f"   • {model['model']}: {model['progress']['current']}/{model['progress']['total']}")
                    print(f"     ({model['progress']['percent']}%, PID: {model['pid']})")
                else:
                    print(f"   • {model['model']}: {model['progress']['status']}")

        # Статус GPU
        print("\n🔥 GPU СТАТУС:")
        gpus = self.get_gpu_status()
        for gpu in gpus:
            if "error" not in gpu:
                mem_pct = round(float(gpu['memory_used'].replace(' MiB', '')) /
                             float(gpu['memory_total'].replace(' MiB', '')) * 100, 1)
                print(f"   GPU {gpu['index']}: {gpu['utilization']} утилизации, {mem_pct}% памяти ({gpu['memory_used']} / {gpu['memory_total']})")

        # Статус карточек
        print("\n📊 СТАТУС КАРТОЧЕК:")
        cards = self.get_cards_status()

        models_with_data = []
        models_empty_quality = []

        for model_name, card_data in cards.items():
            if not card_data['quality_empty']:
                models_with_data.append(model_name)
            else:
                models_empty_quality.append(model_name)

        print(f"   С данными: {len(models_with_data)} моделей")
        print(f"   Quality пустое: {len(models_empty_quality)} моделей")

        if models_empty_quality:
            print(f"   Ожидают данные: {', '.join(models_empty_quality)}")

        # Очередность
        print("\n📋 ОЧЕРЕДНОСТЬ:")
        queue = ["qwen2.5-vl-7b", "glm-4-9b-chat", "ollama-qwen3-8b"]
        for i, model in enumerate(queue, 1):
            print(f"   {i}. {model}")

        print("\n" + "=" * 80)


def main():
    tracker = ProgressTracker()

    try:
        while True:
            tracker.print_dashboard()
            time.sleep(300)  # Обновление каждые 5 минут
    except KeyboardInterrupt:
        print("\nDashboard остановлен")


if __name__ == "__main__":
    main()
