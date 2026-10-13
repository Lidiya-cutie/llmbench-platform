"""Интеграция vision и agent тестов в основной pipeline.

Безопасная интеграция без изменения существующих модулей.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime

from llmbench.vision_agent_tests import VisionAgentTester
from llmbench.modelcard import VISION_BENCHES, is_vision_capable


class VisionAgentIntegrator:
    """Интеграция vision и agent возможностей в систему."""

    def __init__(self):
        self.tester = VisionAgentTester()
        self.results = {}

    def analyze_all_models(self, card_dir: str = "out") -> Dict[str, Any]:
        """Анализ всех моделей на vision/agent возможности."""

        cards = list(Path(card_dir).glob("card_*.json"))

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(cards),
            "vision_models": [],
            "tool_calling_models": [],
            "recommendations": []
        }

        for card_path in cards:
            try:
                with open(card_path) as f:
                    card = json.load(f)

                model_name = card.get('name', 'unknown')
                modalities = card.get('modalities', '')
                micro = card.get('micro', {})
                vision = card.get('vision') or {}
                quality = card.get('quality') or {}

                if is_vision_capable(modalities):
                    tested = any(
                        isinstance(vision.get(k), (int, float))
                        or isinstance(quality.get(k), (int, float))
                        for k in ("docvqa", "ocrbench")
                    )
                    analysis["vision_models"].append({
                        "model": model_name,
                        "modalities": modalities,
                        "tested": tested,
                        "status": vision.get("status"),
                    })
                    if not tested:
                        analysis["recommendations"].append(
                            f"{model_name}: протестировать DocVQA/OCRBench "
                            f"(scripts/run_vlm_docvqa_ocr.sh)"
                        )

                tool_call_ok = micro.get('tool_call_ok', None)
                if tool_call_ok is not None and tool_call_ok > 0:
                    analysis["tool_calling_models"].append({
                        "model": model_name,
                        "tool_call_ok": tool_call_ok
                    })

                if tool_call_ok is None:
                    analysis["recommendations"].append(
                        f"{model_name}: протестировать BFCL"
                    )

            except Exception as e:
                print(f"Ошибка анализа {card_path}: {e}")

        self.results["all_models_analysis"] = analysis
        return analysis

    def get_vision_ready_models(self, card_dir: str = "out") -> List[str]:
        """Получить список моделей с vision поддержкой."""

        vision_models = []
        cards = list(Path(card_dir).glob("card_*.json"))

        for card_path in cards:
            try:
                with open(card_path) as f:
                    card = json.load(f)

                modalities = card.get('modalities', '')
                if is_vision_capable(modalities):
                    vision_models.append(card.get('name', 'unknown'))

            except Exception:
                pass

        return vision_models

    def get_agent_ready_models(self, card_dir: str = "out") -> List[str]:
        """Получить список моделей с agent поддержкой."""

        agent_models = []
        cards = list(Path(card_dir).glob("card_*.json"))

        for card_path in cards:
            try:
                with open(card_path) as f:
                    card = json.load(f)

                micro = card.get('micro', {})
                tool_call_ok = micro.get('tool_call_ok', None)

                if tool_call_ok is not None and tool_call_ok > 0:
                    agent_models.append(card.get('name', 'unknown'))

            except:
                pass

        return agent_models

    def update_model_card(self, model_name: str, new_metrics: Dict[str, Any],
                          card_dir: str = "out") -> bool:
        """Безопасное обновление карточки модели новыми метриками."""

        card_path = Path(card_dir) / f"card_{model_name}.json"

        if not card_path.exists():
            return False

        try:
            # Создание backup
            backup_path = card_path.with_suffix(
                f".json.backup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )

            with open(card_path) as f:
                card = json.load(f)

            # Backup
            with open(backup_path, 'w') as f:
                json.dump(card, f, indent=2)

            # Vision-метрики → card.vision; в quality только реальные float benches
            vision = dict(card.get("vision") or {})
            quality = dict(card.get("quality") or {})
            for junk in (
                "docvqa_score",
                "ocrbench_score",
                "chartqa_score",
                "mmmu_score",
                "vision_tested",
                "vision_test_date",
                "last_updated",
            ):
                quality.pop(junk, None)

            for k, v in new_metrics.items():
                key = k.replace("_score", "")
                if key in VISION_BENCHES and isinstance(v, (int, float)) and float(v) > 0:
                    vision[key] = float(v)
                    quality[key] = float(v)
                elif key in ("status", "note"):
                    vision[key] = v
                elif key.startswith("vision_"):
                    vision[key.replace("vision_", "", 1)] = v

            if any(isinstance(vision.get(k), (int, float)) for k in VISION_BENCHES):
                vision["status"] = vision.get("status") or "done"
            vision["updated_at"] = datetime.now().isoformat()
            card["vision"] = vision
            card["quality"] = quality

            with open(card_path, 'w') as f:
                json.dump(card, f, indent=2)

            return True

        except Exception as e:
            print(f"Ошибка обновления {model_name}: {e}")
            return False

    def generate_summary_report(self, output_path: Optional[str] = None) -> str:
        """Генерация сводного отчета по vision/agent возможностям."""

        if output_path is None:
            output_path = "out/vision_agent_summary.json"

        analysis = self.analyze_all_models()

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": analysis,
            "vision_models_count": len(analysis["vision_models"]),
            "agent_models_count": len(analysis["tool_calling_models"]),
            "recommendations_count": len(analysis["recommendations"]),
            "details": {
                "vision_models": analysis["vision_models"],
                "agent_models": analysis["tool_calling_models"],
                "recommendations": analysis["recommendations"]
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        return output_path


if __name__ == "__main__":
    import sys

    integrator = VisionAgentIntegrator()

    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        print("АНАЛИЗ ВСЕХ МОДЕЛЕЙ:")
        print("=" * 30)

        analysis = integrator.analyze_all_models()

        print(f"Всего моделей: {analysis['total_models']}")
        print(f"Vision моделей: {len(analysis['vision_models'])}")
        print(f"Agent моделей: {len(analysis['tool_calling_models'])}")
        print(f"Рекомендаций: {len(analysis['recommendations'])}")

        if analysis['recommendations']:
            print("\nРЕКОМЕНДАЦИИ:")
            for rec in analysis['recommendations'][:5]:
                print(f"  - {rec}")

    elif len(sys.argv) > 1 and sys.argv[1] == "--report":
        report_path = integrator.generate_summary_report()
        print(f"Отчет сохранен: {report_path}")

    else:
        print("Используйте --analyze для анализа или --report для генерации отчета")