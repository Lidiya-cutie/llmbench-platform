"""
История оценок моделей - хранение и тренды.

Backward compatible: НЕ прерывает evaluate_model.py
Используется как post-processing для импорта карточек.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EvalRun:
    """Прогон оценки модели"""
    model_key: str
    timestamp: datetime
    suite: Optional[str]
    quality: Dict[str, float]
    perf: Dict[str, float]
    hardware: Dict[str, float]
    cost: Dict[str, float]
    micro: Dict[str, float]

    def to_dict(self) -> dict:
        """Конвертация в словарь"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class HistoryStore:
    """Хранилище истории оценок (SQLite)"""

    def __init__(self, db_path: str = "out/history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Создание таблиц (если не существуют)"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                suite TEXT,
                quality_json TEXT,
                perf_json TEXT,
                hardware_json TEXT,
                cost_json TEXT,
                micro_json TEXT,
                UNIQUE(model_key, timestamp)
            )
        """)
        conn.commit()
        conn.close()

    def add_run(self, run: EvalRun) -> bool:
        """Добавление прогона (upsert по model_key + timestamp)"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO eval_runs
                (model_key, timestamp, suite, quality_json, perf_json, hardware_json, cost_json, micro_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.model_key,
                run.timestamp.isoformat(),
                run.suite,
                json.dumps(run.quality),
                json.dumps(run.perf),
                json.dumps(run.hardware),
                json.dumps(run.cost),
                json.dumps(run.micro)
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Ошибка добавления прогона: {e}")
            return False
        finally:
            conn.close()

    def get_latest(self, model_key: str) -> Optional[EvalRun]:
        """Получение последнего прогона модели"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute("""
            SELECT * FROM eval_runs
            WHERE model_key = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (model_key,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_run(row)

    def get_trend(self, model_key: str, days: int = 30) -> List[EvalRun]:
        """Получение тренда качества за период"""
        conn = sqlite3.connect(self.db_path)
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None).timestamp() - (days * 86400)

        cur = conn.execute("""
            SELECT * FROM eval_runs
            WHERE model_key = ? AND datetime(timestamp) > datetime(?, 'unixepoch')
            ORDER BY timestamp ASC
        """, (model_key, cutoff))

        rows = cur.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def get_all_models(self) -> List[str]:
        """Получение списка всех моделей"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute("SELECT DISTINCT model_key FROM eval_runs")
        models = [row[0] for row in cur.fetchall()]
        conn.close()
        return sorted(models)

    def _row_to_run(self, row: tuple) -> EvalRun:
        """Конвертация строки БД в EvalRun"""
        return EvalRun(
            model_key=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            suite=row[3],
            quality=json.loads(row[4]),
            perf=json.loads(row[5]),
            hardware=json.loads(row[6]),
            cost=json.loads(row[7]),
            micro=json.loads(row[8])
        )


def import_card(card_path: str, store: HistoryStore) -> bool:
    """Импорт карточки в историю (post-processing)"""
    try:
        with open(card_path, "r", encoding="utf-8") as f:
            card = json.load(f)

        run = EvalRun(
            model_key=card.get("model_key", card.get("name", "")),
            timestamp=datetime.fromisoformat(card.get("generated_at", datetime.now(timezone.utc).isoformat())),
            suite=card.get("eval_suite"),
            quality=card.get("quality", {}),
            perf=card.get("perf", {}),
            hardware=card.get("hardware", {}),
            cost=card.get("cost", {}),
            micro=card.get("micro", {})
        )

        return store.add_run(run)
    except Exception as e:
        print(f"Ошибка импорта {card_path}: {e}")
        return False


def import_all_cards(out_dir: str = "out", db_path: str = "out/history.db") -> int:
    """Импорт всех карточек из директории"""
    store = HistoryStore(db_path)
    imported = 0

    for card_path in Path(out_dir).glob("card_*.json"):
        if import_card(str(card_path), store):
            imported += 1
            print(f"✓ Импортирована: {card_path.name}")
        else:
            print(f"✗ Ошибка: {card_path.name}")

    return imported


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="История оценок моделей")
    parser.add_argument("--import-cards", action="store_true", help="Импортировать все карточки")
    parser.add_argument("--out-dir", default="out", help="Директория с карточками")
    parser.add_argument("--db-path", default="out/history.db", help="Путь к БД")
    parser.add_argument("--list", action="store_true", help="Список всех моделей")
    parser.add_argument("--trend", type=str, help="Тренд модели")
    parser.add_argument("--days", type=int, default=30, help="Период тренда в днях")

    args = parser.parse_args()

    if args.import_cards:
        count = import_all_cards(args.out_dir, args.db_path)
        print(f"Импортировано карточек: {count}")

    elif args.list:
        store = HistoryStore(args.db_path)
        models = store.get_all_models()
        print("Модели в истории:")
        for model in models:
            print(f"  - {model}")

    elif args.trend:
        store = HistoryStore(args.db_path)
        runs = store.get_trend(args.trend, args.days)
        print(f"Тренд {args.trend} за {args.days} дней:")
        for run in runs:
            quality = run.quality.get("mmlu_pro", "N/A")
            print(f"  {run.timestamp}: mmlu_pro={quality}")
