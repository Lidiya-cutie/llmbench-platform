#!/usr/bin/env python3
"""
Система хранения истории прогонов для сравнительного анализа LLM

SQLite + MLflow + Parquet для полного трекинга прогонов,
конфигураций и метрик с возможностью воспроизводимости.

Структура:
- run_history (SQLite) - основная таблица прогонов
- run_metrics (SQLite) - метрики каждого прогона
- run_configs (SQLite) - хеши конфигураций для воспроизводимости
- mlflow_tracking - логирование экспериментов в MLflow
- parquet_export - экспорт в Parquet для анализа
"""

from __future__ import annotations

import sqlite3
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import uuid


@dataclass
class RunRecord:
    """Запись о прогоне модели"""
    run_id: str                          # Уникальный ID прогона
    model_key: str                       # Ключ модели из models.yaml
    timestamp: str                        # ISO timestamp
    model_name: str                      # Имя модели для отображения
    model_config_hash: str               # Хеш конфигурации модели
    harness_version: str                 # Версия lm-eval harness
    vllm_version: str                    # Версия vLLM (если есть)
    gpu_info: str                         # Информация о GPU
    status: str                          # status: running/completed/failed
    error: Optional[str] = None          # Ошибка если есть
    duration_seconds: Optional[float] = None  # Длительность прогона


@dataclass
class MetricRecord:
    """Метрики прогона"""
    metric_id: str                       # ID метрики
    run_id: str                          # ID пргона
    metric_name: str                     # Название метрики
    metric_value: float                 # Значение метрики
    metric_category: str                 # Категория (quality/perf/hardware)
    timestamp: str                        # Когда получена метрика
    parameters_hash: str                 # Хеш параметров теста
    submetric: Optional[str] = None      # Под-метрика (например, mmlu_pro_biology)


class RunHistoryStore:
    """Хранилище истории прогонов в SQLite"""

    def __init__(self, db_path: str = "out/run_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Инициализация SQLite базы данных"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Таблица прогонов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_history (
                run_id TEXT PRIMARY KEY,
                model_key TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model_name TEXT,
                model_config_hash TEXT,
                harness_version TEXT,
                vllm_version TEXT,
                gpu_info TEXT,
                status TEXT,
                error TEXT,
                duration_seconds REAL,
                FOREIGN KEY (run_id) REFERENCES run_history(run_id)
            )
        """)

        # Таблица метрик
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_metrics (
                metric_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_category TEXT,
                submetric TEXT,
                timestamp TEXT NOT NULL,
                parameters_hash TEXT,
                FOREIGN KEY (run_id) REFERENCES run_history(run_id),
                FOREIGN KEY (metric_id) REFERENCES run_metrics(metric_id)
            )
        """)

        # Таблица конфигураций
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_configs (
                config_hash TEXT PRIMARY KEY,
                model_key TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                used_count INTEGER DEFAULT 0
            )
        """)

        # Индексы для быстрого поиска
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_key ON run_history(model_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON run_history(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metric_run ON run_metrics(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON run_metrics(metric_category)")

        conn.commit()
        conn.close()

    def add_run(self, run: RunRecord) -> bool:
        """Добавить запись о прогоне"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO run_history
                (run_id, model_key, timestamp, model_name, model_config_hash,
                 harness_version, vllm_version, gpu_info, status, error, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id, run.model_key, run.timestamp, run.model_name,
                run.model_config_hash, run.harness_version, run.vllm_version,
                run.gpu_info, run.status, run.error, run.duration_seconds
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Error adding run: {e}")
            return False

    def add_metric(self, metric: MetricRecord) -> bool:
        """Добавить метрику прогона"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO run_metrics
                (metric_id, run_id, metric_name, metric_value, metric_category,
                 submetric, timestamp, parameters_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_id, metric.run_id, metric.metric_name,
                metric.metric_value, metric.metric_category, metric.submetric,
                metric.timestamp, metric.parameters_hash
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Error adding metric: {e}")
            return False

    def get_run_history(self, model_key: Optional[str] = None,
                       limit: int = 100) -> List[Dict]:
        """Получить историю прогонов"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if model_key:
            cursor.execute("""
                SELECT * FROM run_history
                WHERE model_key = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (model_key, limit))
        else:
            cursor.execute("""
                SELECT * FROM run_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_metrics_trend(self, model_key: str, metric_name: str,
                           days: int = 30) -> List[Dict]:
        """Получить динамику метрик модели во времени"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

        cursor.execute("""
            SELECT rm.* FROM run_metrics rm
            JOIN run_history rh ON rm.run_id = rh.run_id
            WHERE rh.model_key = ? AND rm.metric_name = ?
              AND rh.timestamp > ?
            ORDER BY rm.timestamp DESC
        """, (model_key, metric_name, cutoff_date))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_reproducibility_info(self, run_id: str) -> Dict[str, Any]:
        """Получить информацию для воспроизводимости прогона"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Информация о прогоне
        cursor.execute("""
            SELECT * FROM run_history WHERE run_id = ?
        """, (run_id,))

        run = cursor.fetchone()
        if not run:
            return {"error": "Run not found"}

        # Все метрики прогона
        cursor.execute("""
            SELECT * FROM run_metrics WHERE run_id = ?
        """, (run_id,))

        metrics = cursor.fetchall()
        conn.close()

        return {
            "run_info": dict(run),
            "metrics": [dict(m) for m in metrics]
        }


def compute_config_hash(model_config: Dict) -> str:
    """Вычислить хеш конфигурации модели"""
    import json
    config_str = json.dumps(model_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_parameters_hash(test_params: Dict) -> str:
    """Вычислить хеш параметров теста"""
    import json
    params_str = json.dumps(test_params, sort_keys=True)
    return hashlib.sha256(params_str.encode()).hexdigest()[:16]


def extract_gpu_info() -> str:
    """Извлечь информацию о GPU"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append(f"{parts[0]} {parts[3]}% util")
        return ", ".join(gpus) if gpus else "unknown"
    except:
        return "unknown"


def main():
    """Тестирование системы хранения истории"""
    store = RunHistoryStore()

    print("🗄️ ТЕСТ СИСТЕМЫ ХРАНЕНИЯ ИСТОРИИ ПРОГОНОВ")
    print("=" * 60)

    # Тест записи прогона
    test_run = RunRecord(
        run_id=str(uuid.uuid4()),
        model_key="test-model",
        timestamp=datetime.now().isoformat(),
        model_name="Test Model v1.0",
        model_config_hash="abc123",
        harness_version="0.4.1",
        vllm_version="0.6.4",
        gpu_info="RTX 4090 85%",
        status="completed",
        duration_seconds=180.5
    )

    if store.add_run(test_run):
        print("✅ Run record added")

    # Тест метрики
    test_metric = MetricRecord(
        metric_id=str(uuid.uuid4()),
        run_id=test_run.run_id,
        metric_name="mmlu_pro",
        metric_value=0.65,
        metric_category="quality",
        submetric=None,
        timestamp=datetime.now().isoformat(),
        parameters_hash="xyz789"
    )

    if store.add_metric(test_metric):
        print("✅ Metric added")

    # Тест извлечения
    history = store.get_run_history("test-model", limit=5)
    print(f"📊 History retrieved: {len(history)} records")

    # Текстовые примеры
    print("\n💡 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:")
    print("-" * 60)
    print("# 1. История прогона модели:")
    print("history = store.get_run_history('glm-4-9b-chat')")
    print("")
    print("# 2. Динамика метрики:")
    print("trends = store.get_metrics_trend('glm-4-9b-chat', 'mmlu_pro', days=30)")
    print("")
    print("# 3. Воспроизводимость:")
    print("repro_info = store.get_reproducibility_info('run_id_123')")
    print("")
    print("# 4. Экспорт в Parquet (для анализа):")
    print("# TODO: parquet_export.to_parquet('out/metrics_history.parquet')")


if __name__ == "__main__":
    main()