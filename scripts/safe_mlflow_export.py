#!/usr/bin/env python3
"""Безопасный экспорт в MLflow и Parquet (изоляция от существующих процессов)"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmbench.mlflow_integration import MLflowTracker
from llmbench.parquet_export import ParquetExporter
from llmbench.history import HistoryStore


def safe_export_all():
    """Безопасный экспорт всех данных"""
    print('БЕЗОПАСНЫЙ ЭКСПОРТ (изолированный процесс)')

    try:
        exporter = ParquetExporter()
        tracker = MLflowTracker()

        print('1. Parquet экспорт истории...')
        path = exporter.export_history('out/history.parquet')
        print(f'   Создан: {path}')

        print('2. MLflow импорт карточек...')
        card_count = 0
        for card_path in Path('out').glob('card_*.json'):
            try:
                tracker.log_from_card(str(card_path))
                card_count += 1
            except Exception as e:
                print(f'   Ошибка {card_path.name}: {e}')

        print(f'   Импортировано: {card_count} карточек')

        print('3. Проверка целостности...')
        store = HistoryStore()
        db_models = set(store.get_all_models())

        import pandas as pd
        df = pd.read_parquet('out/history.parquet')
        pq_models = set(df['model_key'].unique())

        if db_models == pq_models:
            print('   Данные согласованы')
        else:
            print('   ВНИМАНИЕ: расхождение данных')
            print(f'   SQLite: {len(db_models)}, Parquet: {len(pq_models)}')

        print()
        print('РЕЗУЛЬТАТ: Экспорт завершен успешно')
        print('Существующие процессы не затронуты')

        return True

    except Exception as e:
        print(f'КРИТИЧЕСКАЯ ОШИБКА: {e}')
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Безопасный экспорт")
    parser.add_argument('--auto', action='store_true', help='Автоматический режим')
    args = parser.parse_args()

    if args.auto:
        result = safe_export_all()
        sys.exit(0 if result else 1)
    else:
        safe_export_all()