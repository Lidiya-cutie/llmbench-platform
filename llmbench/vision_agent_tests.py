#!/usr/bin/env python3
"""Безопасное тестирование vision и agent возможностей моделей.

Изолированный модуль для тестирования мультимодальных и агентных возможностей
без влияния на существующие процессы.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
import os

# Добавляем родительскую директорию в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmbench.bfcl_integration import check_bfcl_available, run_bfcl, parse_bfcl_for_card, BFCLResult
from llmbench.modelcard import is_vision_capable


class VisionAgentTester:
    """Безопасное тестирование vision и agent возможностей."""

    def __init__(self, output_dir: str = "out/vision_agent_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def test_vision_capability(self, model_name: str, model_config: Dict) -> Dict[str, Any]:
        """Тестирование vision возможности модели."""

        print(f"Тестирование vision для {model_name}...")

        modalities = model_config.get('modalities', '')
        if not is_vision_capable(modalities):
            return {
                "model": model_name,
                "vision_supported": False,
                "timestamp": datetime.now().isoformat(),
                "error": "Model does not support vision"
            }

        kit_ok = self._check_vlmeval_available()
        result = {
            "model": model_name,
            "vision_supported": True,
            "vlmeval_available": kit_ok,
            "timestamp": datetime.now().isoformat(),
            "status": "adapter_pending" if kit_ok else "kit_missing",
            "note": (
                "Реальный DocVQA/OCRBench: LLMBENCH_VL_EXECUTE=1 "
                "scripts/run_vlm_docvqa_ocr.sh (не пишет fake 0.0 в quality)"
            ),
            "benches": ["docvqa", "ocrbench", "chartqa", "mmmu"],
        }
        if not kit_ok:
            result["error"] = "VLMEvalKit not available / .venv-vlmeval missing"

        self.results[f"vision_{model_name}"] = result
        return result

    def test_agent_capability(self, model_name: str, base_url: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Тестирование agent возможности модели через BFCL."""

        print(f"Тестирование agent capability для {model_name}...")

        # Проверка BFCL доступности
        if not check_bfcl_available():
            return {
                "model": model_name,
                "agent_capability": "unknown",
                "bfcl_available": False,
                "timestamp": datetime.now().isoformat(),
                "error": "BFCL not installed"
            }

        try:
            # Запуск BFCL теста
            bfcl_result = run_bfcl(base_url, model_name)

            result = {
                "model": model_name,
                "agent_capability": "tested",
                "bfcl_available": True,
                "bfcl_overall_accuracy": bfcl_result.overall_accuracy,
                "bfcl_ast_summary": bfcl_result.ast_summary,
                "bfcl_multi_turn": bfcl_result.multi_turn,
                "timestamp": datetime.now().isoformat(),
                "status": "tested"
            }

            if bfcl_result.detail and "error" in bfcl_result.detail:
                result["error"] = bfcl_result.detail["error"]
                result["status"] = "error"

            self.results[f"agent_{model_name}"] = result
            return result

        except Exception as e:
            return {
                "model": model_name,
                "agent_capability": "error",
                "bfcl_available": True,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def test_model_card_vision(self, card_path: str) -> Dict[str, Any]:
        """Тестирование vision из карточки модели."""

        try:
            with open(card_path) as f:
                card = json.load(f)

            model_name = card.get('name', 'unknown')

            # Проверка текущих метрик
            micro = card.get('micro', {})
            current_tool_call = micro.get('tool_call_ok', None)

            modalities = card.get('modalities', '')
            vision_supported = is_vision_capable(modalities)

            result = {
                "model": model_name,
                "vision_supported": vision_supported,
                "current_tool_call_ok": current_tool_call,
                "modalities": modalities,
                "timestamp": datetime.now().isoformat(),
                "card_path": card_path
            }

            # Рекомендации по тестированию
            recommendations = []

            if vision_supported and not current_tool_call:
                recommendations.append("Test vision capabilities")

            if current_tool_call is None:
                recommendations.append("Test tool-calling capabilities")

            result["recommendations"] = recommendations

            self.results[f"card_analysis_{model_name}"] = result
            return result

        except Exception as e:
            return {
                "model": "unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _check_vlmeval_available(self) -> bool:
        """Проверка доступности VLMEvalKit."""
        try:
            vlmeval_path = Path("third_party/VLMEvalKit")
            return vlmeval_path.exists()
        except:
            return False

    def save_results(self, filename: Optional[str] = None) -> str:
        """Сохранение результатов тестирования."""

        if filename is None:
            filename = f"vision_agent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        return str(output_path)

    def get_summary(self) -> Dict[str, Any]:
        """Получение сводки результатов тестирования."""

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "vision_tests": len([k for k in self.results.keys() if k.startswith("vision_")]),
            "agent_tests": len([k for k in self.results.keys() if k.startswith("agent_")]),
            "card_analyses": len([k for k in self.results.keys() if k.startswith("card_analysis_")])
        }

        # Статистика по статусам
        statuses = {}
        for key, result in self.results.items():
            status = result.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1

        summary["statuses"] = statuses

        return summary


def test_single_model_safe(model_name: str, card_path: str) -> Dict[str, Any]:
    """Безопасное тестирование одной модели."""

    tester = VisionAgentTester()

    # Анализ карточки
    card_result = tester.test_model_card_vision(card_path)

    # Если vision поддерживается, предлагаем протестировать
    if card_result.get("vision_supported"):
        print(f"{model_name}: Vision поддерживается, можно протестировать DocVQA/OCRBench")
        vision_result = tester.test_vision_capability(model_name, card_result)

    # Если tool-calling не протестирован, предлагаем протестировать BFCL
    if card_result.get("current_tool_call_ok") is None:
        print(f"{model_name}: Tool-calling не протестирован, можно запустить BFCL")

    # Сохранение результатов
    result_path = tester.save_results(f"{model_name}_test_results.json")

    return {
        "model": model_name,
        "card_analysis": card_result,
        "results_saved": result_path,
        "summary": tester.get_summary()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Безопасное тестирование vision и agent возможностей")
    parser.add_argument("--model", help="Название модели для тестирования")
    parser.add_argument("--card", help="Путь к карточке модели")
    parser.add_argument("--base-url", help="Base URL для BFCL тестирования")
    parser.add_argument("--test-vision", action="store_true", help="Тестировать vision возможности")
    parser.add_argument("--test-agent", action="store_true", help="Тестировать agent возможности")
    parser.add_argument("--analyze-cards", action="store_true", help="Анализировать все карточки")

    args = parser.parse_args()

    if args.analyze_cards:
        print("АНАЛИЗ ВСЕХ КАРТОЧЕК:")
        print("=" * 30)

        tester = VisionAgentTester()

        for card_path in Path("out").glob("card_*.json"):
            result = tester.test_model_card_vision(str(card_path))
            model = result.get("model", "unknown")

            print(f"\n{model}:")
            print(f"  Vision: {'ДА' if result.get('vision_supported') else 'НЕТ'}")
            print(f"  Tool-calling: {result.get('current_tool_call_ok', 'не тестировался')}")

            if result.get("recommendations"):
                print(f"  Рекомендации: {', '.join(result['recommendations'])}")

        # Сохранение сводки
        summary = tester.get_summary()
        result_path = tester.save_results("all_cards_analysis.json")
        print(f"\nРезультаты сохранены: {result_path}")
        print(f"Всего проанализировано: {summary['card_analyses']} карточек")

    elif args.model and args.card:
        print(f"ТЕСТИРОВАНИЕ {args.model}:")
        print("=" * 30)

        result = test_single_model_safe(args.model, args.card)

        print(f"\nРЕЗУЛЬТАТ:")
        print(f"  Модель: {result['model']}")
        print(f"  Vision поддержка: {'ДА' if result['card_analysis'].get('vision_supported') else 'НЕТ'}")
        print(f"  Tool-calling: {result['card_analysis'].get('current_tool_call_ok', 'не тестировался')}")
        print(f"  Результаты сохранены: {result['results_saved']}")

    else:
        print("Используйте --analyze-cards для анализа всех карточек")
        print("Или укажите --model и --card для тестирования одной модели")
        parser.print_help()