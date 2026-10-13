"""Интеграция с MLflow для логирования экспериментов"""

import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional
import json
import hashlib


class MLflowTracker:
    """MLflow логирование для экспериментов сравнения моделей"""

    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "llmbench"):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name
        self._ensure_experiment()

    def _ensure_experiment(self):
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.experiment_name)

    def log_model_run(self, model_key: str, metrics: Dict[str, float],
                    params: Dict[str, Any], artifacts: Optional[list] = None):
        """Логирование прогона модели"""
        with mlflow.start_run():
            mlflow.set_tag("model_key", model_key)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            if artifacts:
                for artifact_path in artifacts:
                    if Path(artifact_path).exists():
                        mlflow.log_artifact(artifact_path)

    def log_from_card(self, card_path: str):
        """Логирование из карточки модели"""
        with open(card_path, 'r') as f:
            card = json.load(f)

        model_key = card.get('name', 'unknown')

        params = {
            'model_config_hash': self._compute_hash(card),
            'engine': card.get('engine', 'unknown'),
            'timestamp': card.get('generated_at', ''),
            'quant_format': card.get('quant_format', ''),
            'context_max': card.get('context_max', 0)
        }

        metrics = {}
        for category in ['quality', 'perf', 'hardware', 'cost', 'micro']:
            category_metrics = card.get(category, {})
            for key, value in category_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"{category}_{key}"] = value

        artifacts = [card_path] if Path(card_path).exists() else []

        self.log_model_run(model_key, metrics, params, artifacts)

    def _compute_hash(self, data: Dict) -> str:
        """Вычисление хеша для конфигурации"""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def setup_mlflow_ui(port: int = 5000):
    """Запуск MLflow UI"""
    import subprocess
    subprocess.run(['mlflow', 'ui', '--port', str(port)])


if __name__ == "__main__":
    tracker = MLflowTracker()
    print("MLflow integration initialized")
    print(f"Experiment: {tracker.experiment_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")