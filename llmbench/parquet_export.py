"""Экспорт истории в Parquet формат для анализа"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime


class ParquetExporter:
    """Экспорт данных истории в Parquet"""

    def __init__(self, db_path: str = "out/history.db"):
        self.db_path = db_path

    def export_history(self, output_path: str = "out/history_export.parquet"):
        """Экспорт полной истории"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)

        df_runs = pd.read_sql_query("SELECT * FROM eval_runs", conn)
        conn.close()

        df_runs['timestamp'] = pd.to_datetime(df_runs['timestamp'], format='ISO8601')

        table = pa.Table.from_pandas(df_runs)
        pq.write_table(table, output_path)

        return output_path

    def export_metrics_by_model(self, model_key: str, output_path: Optional[str] = None):
        """Экспорт метрик конкретной модели"""
        import sqlite3

        if output_path is None:
            output_path = f"out/{model_key}_metrics.parquet"

        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT * FROM eval_runs
            WHERE model_key = ?
            ORDER BY timestamp DESC
        """

        df = pd.read_sql_query(query, conn, params=(model_key,))
        conn.close()

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')

        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)

        return output_path

    def export_time_series(self, days: int = 30, output_path: str = "out/timeseries.parquet"):
        """Экспорт временных рядов для анализа динамики"""
        import sqlite3
        from datetime import timedelta

        conn = sqlite3.connect(self.db_path)

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        query = """
            SELECT model_key, timestamp,
                   quality_json, perf_json, hardware_json
            FROM eval_runs
            WHERE timestamp > ?
            ORDER BY model_key, timestamp ASC
        """

        df = pd.read_sql_query(query, conn, params=(cutoff,))
        conn.close()

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')

        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)

        return output_path

    def get_parquet_info(self, parquet_path: str) -> dict:
        """Информация о Parquet файле"""
        parquet_file = pq.ParquetFile(parquet_path)

        metadata = parquet_file.metadata
        schema = parquet_file.schema_arrow

        return {
            'rows': metadata.num_rows,
            'columns': len(schema),
            'size_bytes': metadata.serialized_size,
            'schema': [field.name for field in schema]
        }


if __name__ == "__main__":
    exporter = ParquetExporter()

    print("ПАРКЕТ ЭКСПОРТЕР")
    print("=" * 30)

    try:
        path = exporter.export_history()
        print(f"История экспортирована: {path}")

        info = exporter.get_parquet_info(path)
        print(f"Строк: {info['rows']}, Колонок: {info['columns']}")
    except Exception as e:
        print(f"Ошибка: {e}")