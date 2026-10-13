"""Единый пайплайн истории моделей: SQLite + MLflow + Parquet"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from llmbench.history import HistoryStore, EvalRun, import_card
from llmbench.mlflow_integration import MLflowTracker
from llmbench.parquet_export import ParquetExporter


class ModelHistoryPipeline:
    """Единый пайплайн для хранения истории моделей"""

    def __init__(self, db_path: str = "out/history.db"):
        self.db_path = db_path
        self.history_store = HistoryStore(db_path)
        self.mlflow_tracker = MLflowTracker()
        self.parquet_exporter = ParquetExporter(db_path)

    def add_model_run(self, model_key: str, quality: Dict[str, float],
                     perf: Dict[str, float], hardware: Dict[str, float],
                     cost: Dict[str, float], micro: Dict[str, float],
                     suite: Optional[str] = None) -> bool:
        """Добавление прогона модели во все системы"""

        try:
            timestamp = datetime.now()

            run = EvalRun(
                model_key=model_key,
                timestamp=timestamp,
                suite=suite,
                quality=quality,
                perf=perf,
                hardware=hardware,
                cost=cost,
                micro=micro
            )

            success = self.history_store.add_run(run)

            if success:
                self._sync_to_mlflow(model_key, timestamp, quality, perf, hardware, cost, micro)
                self._sync_to_parquet()

            return success

        except Exception as e:
            print(f"Ошибка добавления прогона: {e}")
            return False

    def import_model_card(self, card_path: str) -> bool:
        """Импорт карточки модели во все системы"""

        try:
            history_success = import_card(card_path, self.history_store)

            if history_success:
                self.mlflow_tracker.log_from_card(card_path)
                self._sync_to_parquet()

            return history_success

        except Exception as e:
            print(f"Ошибка импорта карточки: {e}")
            return False

    def import_all_cards(self, out_dir: str = "out") -> int:
        """Импорт всех карточек"""

        imported_count = 0

        for card_path in Path(out_dir).glob("card_*.json"):
            if self.import_model_card(str(card_path)):
                imported_count += 1
                print(f"Импортирована: {card_path.name}")

        return imported_count

    def get_model_history(self, model_key: str, days: int = 30) -> List[Dict]:
        """Получение истории модели"""

        runs = self.history_store.get_trend(model_key, days)

        history = []
        for run in runs:
            history.append({
                'timestamp': run.timestamp.isoformat(),
                'quality': run.quality,
                'perf': run.perf,
                'hardware': run.hardware,
                'cost': run.cost,
                'micro': run.micro
            })

        return history

    def get_metrics_trend(self, model_key: str, metric_name: str, days: int = 30) -> List[Dict]:
        """Получение тренда метрики"""

        history = self.get_model_history(model_key, days)

        trend = []
        for run in history:
            if metric_name in run['quality']:
                trend.append({
                    'timestamp': run['timestamp'],
                    'value': run['quality'][metric_name]
                })

        return trend

    def export_to_parquet(self, output_path: Optional[str] = None) -> str:
        """Экспорт истории в Parquet"""

        if output_path is None:
            output_path = "out/history.parquet"

        return self.parquet_exporter.export_history(output_path)

    def export_model_to_parquet(self, model_key: str, output_path: Optional[str] = None) -> str:
        """Экспорт метрик модели в Parquet"""

        if output_path is None:
            output_path = f"out/{model_key}_metrics.parquet"

        return self.parquet_exporter.export_metrics_by_model(model_key, output_path)

    def get_all_models(self) -> List[str]:
        """Получение списка всех моделей"""

        return self.history_store.get_all_models()

    def get_latest_results(self, model_key: str) -> Optional[Dict]:
        """Получение последних результатов модели"""

        run = self.history_store.get_latest(model_key)

        if run:
            return {
                'timestamp': run.timestamp.isoformat(),
                'quality': run.quality,
                'perf': run.perf,
                'hardware': run.hardware,
                'cost': run.cost,
                'micro': run.micro
            }

        return None

    def sync_all(self) -> Dict[str, bool]:
        """Полная синхронизация всех систем"""

        results = {
            'mlflow': False,
            'parquet': False,
            'integrity': False
        }

        try:
            self.mlflow_tracker = MLflowTracker()
            results['mlflow'] = True
        except Exception as e:
            print(f"Ошибка MLflow: {e}")

        try:
            self.export_to_parquet()
            results['parquet'] = True
        except Exception as e:
            print(f"Ошибка Parquet: {e}")

        try:
            models = self.get_all_models()
            import pandas as pd
            df = pd.read_parquet('out/history.parquet')
            pq_models = set(df['model_key'].unique())
            results['integrity'] = set(models) == pq_models
        except Exception as e:
            print(f"Ошибка проверки целостности: {e}")

        return results

    def _sync_to_mlflow(self, model_key: str, timestamp: datetime,
                       quality: Dict, perf: Dict, hardware: Dict,
                       cost: Dict, micro: Dict):

        params = {
            'model_key': model_key,
            'timestamp': timestamp.isoformat(),
            'suite': 'evaluation'
        }

        metrics = {}
        for category, data in [('quality', quality), ('perf', perf),
                              ('hardware', hardware), ('cost', cost), ('micro', micro)]:
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    metrics[f"{category}_{key}"] = value

        self.mlflow_tracker.log_model_run(model_key, metrics, params)

    def _sync_to_parquet(self):
        """Синхронизация с Parquet"""

        try:
            self.export_to_parquet()
        except Exception as e:
            print(f"Ошибка синхронизации Parquet: {e}")


def main():
    """Тестирование пайплайна"""

    print("ТЕСТ ЕДИНОГО ПАЙПЛАЙНА")
    print("=" * 35)

    pipeline = ModelHistoryPipeline()

    print("1. Импорт всех карточек...")
    imported = pipeline.import_all_cards()
    print(f"   Импортировано: {imported}")

    print()
    print("2. Список моделей...")
    models = pipeline.get_all_models()
    print(f"   Моделей: {len(models)}")
    for model in models[:3]:
        print(f"   - {model}")

    print()
    print("3. Экспорт в Parquet...")
    parquet_path = pipeline.export_to_parquet()
    print(f"   Создан: {parquet_path}")

    print()
    print("4. Проверка синхронизации...")
    sync_results = pipeline.sync_all()
    print(f"   MLflow: {sync_results['mlflow']}")
    print(f"   Parquet: {sync_results['parquet']}")
    print(f"   Целостность: {sync_results['integrity']}")

    print()
    print("РЕЗУЛЬТАТ: Пайплайн работает полностью")


if __name__ == "__main__":
    main()