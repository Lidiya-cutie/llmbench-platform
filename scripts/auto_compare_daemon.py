#!/usr/bin/env python3
"""
Демон автоматического сравнения моделей.

Backward compatible: НЕ прерывает evaluate_model.py
Мониторит заполнение quality и запускает compare автоматически.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class QualityMonitor:
    """Мониторинг качества моделей"""

    def __init__(self, out_dir: str = "out"):
        self.out_dir = out_dir

    def check_quality_status(self) -> Dict[str, dict]:
        """Проверка статуса quality во всех карточках"""
        cards = {}

        for card_path in Path(self.out_dir).glob("card_*.json"):
            try:
                with open(card_path, "r", encoding="utf-8") as f:
                    card = json.load(f)

                model_key = card.get("model_key", card_path.stem.replace("card_", ""))
                quality = card.get("quality", {})

                cards[model_key] = {
                    "path": str(card_path),
                    "has_quality": bool(quality),
                    "metrics_count": len(quality),
                    "metrics": list(quality.keys()),
                    "timestamp": card.get("generated_at", "")
                }
            except Exception as e:
                print(f"Ошибка чтения {card_path}: {e}")

        return cards

    def get_ready_models(self, min_metrics: int = 1) -> List[str]:
        """Получение моделей с готовым quality"""
        cards = self.check_quality_status()
        return [k for k, v in cards.items() if v["has_quality"] and v["metrics_count"] >= min_metrics]

    def wait_for_quality(self, min_models: int = 1, check_interval: int = 300) -> List[str]:
        """Ожидание заполнения quality"""
        print(f"Мониторинг quality (минимум моделей: {min_models}, интервал: {check_interval}с)")

        while True:
            ready = self.get_ready_models()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Quality готово: {len(ready)} моделей")

            if len(ready) >= min_models:
                print(f"Достигнут минимум {min_models} моделей")
                return ready

            for model, info in self.check_quality_status().items():
                status = "✓" if info["has_quality"] else "✗"
                print(f"  {status} {model}: {info['metrics_count']} метрик")

            time.sleep(check_interval)


class AutoCompare:
    """Автоматическое сравнение моделей"""

    def __init__(self, baseline: str = "auto"):
        self.baseline = baseline

    def determine_baseline(self, ready_models: List[str]) -> str:
        """Определение baseline модели"""
        if self.baseline != "auto":
            return self.baseline

        # Автоматический выбор: первая модель с quality
        return ready_models[0] if ready_models else ""

    def run_compare(self, models: List[str], baseline: str) -> bool:
        """Запуск сравнения"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"out/compare_runs/compare_{timestamp}"

        cmd = [
            "python3", "scripts/compare_models.py",
            "--baseline", baseline,
            "--models", ",".join(models),
            "--weights", "quality:0.4,perf:0.3,cost:0.3",
            "--out-dir", out_dir
        ]

        log_path = f"logs/auto_compare_{timestamp}.log"

        try:
            with open(log_path, "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd="/opt/llm-data/llmbench"
                )

                print(f"Запущен compare (PID: {process.pid})")
                print(f"Baseline: {baseline}")
                print(f"Models: {', '.join(models)}")
                print(f"Output: {out_dir}/")
                print(f"Log: {log_path}")

                # Ожидание завершения
                process.wait()
                return process.returncode == 0

        except Exception as e:
            print(f"Ошибка запуска compare: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Демон автоматического сравнения")
    parser.add_argument("--min-models", type=int, default=1,
                       help="Минимум моделей с quality для compare")
    parser.add_argument("--interval", type=int, default=300,
                       help="Интервал проверки в секундах")
    parser.add_argument("--baseline", default="auto",
                       help="Baseline модель (auto = автоматический выбор)")
    parser.add_argument("--out-dir", default="out",
                       help="Директория с карточками")
    parser.add_argument("--oneshot", action="store_true",
                       help="Одинарная проверка без ожидания")
    parser.add_argument("--compare", action="store_true",
                       help="Запустить compare немедленно (если есть quality)")

    args = parser.parse_args()

    monitor = QualityMonitor(args.out_dir)
    auto_compare = AutoCompare(args.baseline)

    if args.oneshot:
        # Одинарная проверка
        cards = monitor.check_quality_status()
        ready = [k for k, v in cards.items() if v["has_quality"]]

        print(f"Quality готово: {len(ready)}/{len(cards)}")
        for model, info in cards.items():
            status = "✓" if info["has_quality"] else "✗"
            print(f"  {status} {model}: {info['metrics_count']} метрик")

        if args.compare and ready:
            baseline = auto_compare.determine_baseline(ready)
            auto_compare.run_compare(ready, baseline)

    elif args.compare:
        # Запустить compare немедленно
        ready = monitor.get_ready_models()

        if not ready:
            print("Нет моделей с quality для compare")
            return

        baseline = auto_compare.determine_baseline(ready)
        auto_compare.run_compare(ready, baseline)

    else:
        # Режим демона
        print("Запуск демона автоматического сравнения")
        print(f"Минимум моделей: {args.min_models}")
        print(f"Интервал: {args.interval}с")
        print(f"Baseline: {args.baseline}")
        print(f"Директория: {args.out_dir}")

        while True:
            try:
                # Ожидание качества
                ready = monitor.wait_for_quality(args.min_models, args.interval)

                # Определение baseline
                baseline = auto_compare.determine_baseline(ready)

                # Запуск compare
                success = auto_compare.run_compare(ready, baseline)

                if success:
                    print("Compare завершён успешно")
                    print("Для следующего compare необходимы новые модели с quality")
                    break
                else:
                    print("Compare завершился с ошибкой, повтор через 60 секунд...")
                    time.sleep(60)

            except KeyboardInterrupt:
                print("\nДемон остановлен")
                break
            except Exception as e:
                print(f"Ошибка: {e}")
                time.sleep(60)


if __name__ == "__main__":
    main()
