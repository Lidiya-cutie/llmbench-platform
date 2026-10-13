"""
Интеграция BFCL (Berkeley Function Calling Leaderboard) в llmbench.

Модуль для запуска BFCL бенчмарка и парсинга результатов.
"""

import json
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BFCLResult:
    """Результаты BFCL тестирования."""
    model: str
    overall_accuracy: float = 0.0
    ast_summary: Dict[str, float] = None  # Abstract Syntax Tree metrics
    multi_turn: Dict[str, float] = None     # Multi-turn execution metrics
    detail: Dict = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "overall_accuracy": self.overall_accuracy,
            "ast_summary": self.ast_summary or {},
            "multi_turn": self.multi_turn or {},
            "detail": self.detail or {}
        }


def check_bfcl_available() -> bool:
    """Проверяет доступность BFCL."""
    try:
        result = subprocess.run(
            ["python", "-c", "import bfcl"],
            capture_output=True,
            timeout=10,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def run_bfcl(base_url: str, model: str, **kwargs) -> BFCLResult:
    """
    Запускает BFCL бенчмарк для модели.

    Args:
        base_url: URL модели (OpenAI-совместимый API)
        model: Название модели
        **kwargs: Дополнительные параметры для BFCL

    Returns:
        BFCLResult с результатами тестирования
    """
    if not check_bfcl_available():
        return BFCLResult(
            model=model,
            detail={"error": "BFCL не установлен. Установите: pip install bfcl"}
        )

    try:
        # Формируем команду для запуска BFCL
        cmd = [
            "python", "-m", "bfcl.cli",
            "--model", model,
            "--base_url", base_url,
            "--output_format", "json"
        ]

        # Запускаем BFCL
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=3600,  # 1 час
            text=True
        )

        if result.returncode != 0:
            return BFCLResult(
                model=model,
                detail={"error": f"BFCL execution failed: {result.stderr[:500]}"}
            )

        # Парсим результаты
        try:
            data = json.loads(result.stdout)
            return BFCLResult(
                model=model,
                overall_accuracy=data.get("overall_accuracy", 0.0),
                ast_summary=data.get("ast_summary", {}),
                multi_turn=data.get("multi_turn", {}),
                detail=data
            )
        except json.JSONDecodeError:
            return BFCLResult(
                model=model,
                detail={"error": "Failed to parse BFCL JSON output"}
            )

    except Exception as e:
        return BFCLResult(
            model=model,
            detail={"error": f"BFCL execution error: {str(e)}"}
        )


def get_bfcl_score(result: BFCLResult) -> float:
    """Извлекает общую точность из BFCL результатов."""
    return result.overall_accuracy


def parse_bfcl_for_card(result: BFCLResult) -> Dict[str, float]:
    """
    Парсит BFCL результаты для модельной карточки.

    Returns:
        Словарь с метриками для включения в карточку
    """
    metrics = {
        "bfcl_overall": result.overall_accuracy,
        "bfcl_ast": result.ast_summary.get("accuracy", 0.0) if result.ast_summary else 0.0,
        "bfcl_multi_turn": result.multi_turn.get("accuracy", 0.0) if result.multi_turn else 0.0,
    }

    # Добавляем детальные метрики если доступны
    if result.detail and "categories" in result.detail:
        for category, scores in result.detail["categories"].items():
            if isinstance(scores, dict) and "accuracy" in scores:
                metrics[f"bfcl_{category}"] = scores["accuracy"]

    return metrics


# --------------------------------------------------------------- EOF -------------------------------------------------
# Пример использования:
#
# result = run_bfcl("http://localhost:8080/v1/chat/completions", "qwen:8b")
# if result.detail.get("error"):
#     print(f"Ошибка: {result.detail['error']}")
# else:
#     print(f"BFCL точность: {result.overall_accuracy:.2%}")
#     metrics = parse_bfcl_for_card(result)