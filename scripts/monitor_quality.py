#!/usr/bin/env python3
"""Мониторинг завершения quality метрик в карточках моделей."""

import json
import glob
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

def check_quality_status(out_dir: str = "out") -> Dict[str, dict]:
    """Проверяет статус quality во всех карточках."""
    cards = {}
    for card_path in glob.glob(f"{out_dir}/card_*.json"):
        with open(card_path) as f:
            card = json.load(f)
            model_key = card.get("model_key", card_path)
            quality = card.get("quality", {})
            cards[model_key] = {
                "path": card_path,
                "has_quality": bool(quality),
                "metrics_count": len(quality),
                "metrics": list(quality.keys()) if quality else []
            }
    return cards

def wait_for_quality(min_models: int = 1, check_interval: int = 300,
                     out_dir: str = "out") -> List[str]:
    """Ожидает заполнения quality минимум в min_models карточках."""
    print(f"Мониторинг quality в {out_dir}/ (интервал: {check_interval}с, минимум: {min_models})")

    while True:
        cards = check_quality_status(out_dir)
        ready = [k for k, v in cards.items() if v["has_quality"]]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Quality готов: {len(ready)}/{len(cards)}")

        for model, info in cards.items():
            status = "✓" if info["has_quality"] else "✗"
            metrics = ", ".join(info["metrics"]) if info["metrics"] else "пусто"
            print(f"  {status} {model}: {info['metrics_count']} метрик [{metrics}]")

        if len(ready) >= min_models:
            print(f"\nДостигнут минимум {min_models} моделей с quality")
            return ready

        time.sleep(check_interval)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Мониторинг quality метрик")
    parser.add_argument("--min-models", type=int, default=1,
                       help="Минимум моделей с quality для завершения")
    parser.add_argument("--interval", type=int, default=300,
                       help="Интервал проверки в секундах")
    parser.add_argument("--out-dir", default="out",
                       help="Директория с карточками")
    parser.add_argument("--oneshot", action="store_true",
                       help="Одинарная проверка без ожидания")

    args = parser.parse_args()

    if args.oneshot:
        cards = check_quality_status(args.out_dir)
        ready = [k for k, v in cards.items() if v["has_quality"]]
        print(f"Quality готов: {len(ready)}/{len(cards)}")
        for model, info in cards.items():
            status = "✓" if info["has_quality"] else "✗"
            print(f"  {status} {model}: {info['metrics_count']} метрик")
    else:
        ready = wait_for_quality(args.min_models, args.interval, args.out_dir)
        print(f"\nГотовы к сравнению: {', '.join(ready)}")
