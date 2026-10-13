#!/usr/bin/env python3
"""CLI интерфейс для пайплайна истории моделей"""

import argparse
import sys
from pathlib import Path
from llmbench.pipeline import ModelHistoryPipeline


def cmd_import(args):
    """Импорт карточек"""
    pipeline = ModelHistoryPipeline(args.db_path)

    if args.all:
        count = pipeline.import_all_cards(args.out_dir)
        print(f"Импортировано карточек: {count}")
    elif args.card:
        success = pipeline.import_model_card(args.card)
        print(f"Результат: {'успех' if success else 'ошибка'}")
    else:
        print("Укажите --all или --card")


def cmd_list(args):
    """Список моделей"""
    pipeline = ModelHistoryPipeline(args.db_path)
    models = pipeline.get_all_models()

    print(f"Модели в истории: {len(models)}")
    for model in models:
        print(f"  - {model}")


def cmd_history(args):
    """История модели"""
    pipeline = ModelHistoryPipeline(args.db_path)
    history = pipeline.get_model_history(args.model, args.days)

    print(f"История {args.model} за {args.days} дней:")
    print(f"Записей: {len(history)}")

    for run in history[-3:]:
        print(f"  {run['timestamp']}: mmlu={run['quality'].get('mmlu_pro', 'N/A')}")


def cmd_trend(args):
    """Тренд метрики"""
    pipeline = ModelHistoryPipeline(args.db_path)
    trend = pipeline.get_metrics_trend(args.model, args.metric, args.days)

    print(f"Тренд {args.metric} для {args.model}:")
    for point in trend:
        print(f"  {point['timestamp']}: {point['value']}")


def cmd_export(args):
    """Экспорт в Parquet"""
    pipeline = ModelHistoryPipeline(args.db_path)

    if args.model:
        path = pipeline.export_model_to_parquet(args.model, args.output)
    else:
        path = pipeline.export_to_parquet(args.output)

    print(f"Экспортировано: {path}")


def cmd_sync(args):
    """Синхронизация систем"""
    pipeline = ModelHistoryPipeline(args.db_path)
    results = pipeline.sync_all()

    print("Результаты синхронизации:")
    for system, status in results.items():
        print(f"  {system}: {'OK' if status else 'FAILED'}")

    overall = all(results.values())
    print(f"\nИтого: {'SUCCESS' if overall else 'PARTIAL'}")
    return 0 if overall else 1


def cmd_status(args):
    """Статус системы"""
    pipeline = ModelHistoryPipeline(args.db_path)

    print("СТАТУС ПАЙПЛАЙНА:")
    print("-" * 25)

    models = pipeline.get_all_models()
    print(f"Моделей: {len(models)}")

    if models:
        latest = pipeline.get_latest_results(models[0])
        if latest:
            print(f"Последний прогона: {latest['timestamp']}")

    import os
    files = [
        ('SQLite DB', 'out/history.db'),
        ('Parquet history', 'out/history.parquet'),
        ('MLflow experiments', 'mlruns/')
    ]

    for name, path in files:
        exists = os.path.exists(path)
        print(f"{name}: {'ЕСТЬ' if exists else 'НЕТ'}")


def main():
    parser = argparse.ArgumentParser(description="Пайплайн истории моделей")
    parser.add_argument('--db-path', default='out/history.db', help='Путь к БД')

    subparsers = parser.add_subparsers(dest='command', help='Команды')

    import_parser = subparsers.add_parser('import', help='Импорт данных')
    import_parser.add_argument('--all', action='store_true', help='Все карточки')
    import_parser.add_argument('--card', help='Конкретная карточка')
    import_parser.add_argument('--out-dir', default='out', help='Директория карточек')

    subparsers.add_parser('list', help='Список моделей')

    history_parser = subparsers.add_parser('history', help='История модели')
    history_parser.add_argument('model', help='Ключ модели')
    history_parser.add_argument('--days', type=int, default=30, help='Период')

    trend_parser = subparsers.add_parser('trend', help='Тренд метрики')
    trend_parser.add_argument('model', help='Ключ модели')
    trend_parser.add_argument('metric', help='Название метрики')
    trend_parser.add_argument('--days', type=int, default=30, help='Период')

    export_parser = subparsers.add_parser('export', help='Экспорт в Parquet')
    export_parser.add_argument('--model', help='Конкретная модель')
    export_parser.add_argument('--output', help='Выходной файл')

    subparsers.add_parser('sync', help='Синхронизация систем')
    subparsers.add_parser('status', help='Статус системы')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        'import': cmd_import,
        'list': cmd_list,
        'history': cmd_history,
        'trend': cmd_trend,
        'export': cmd_export,
        'sync': cmd_sync,
        'status': cmd_status
    }

    try:
        return commands[args.command](args)
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())