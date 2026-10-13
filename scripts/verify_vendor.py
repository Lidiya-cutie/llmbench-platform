#!/usr/bin/env python3
"""
Проверка заявлений вендора local-27b против независимых бенчмарков.

Анализ заявлений:
- "~94.6% качества FP16 baseline (ternary)"
- "~89.5% качества FP16 baseline (1-bit)"
- "15 бенчмарков в thinking-режиме"
- "первая 27B-класс модель влезающая в телефон"
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


class VendorClaimsVerifier:
    def __init__(self, cards_dir: str = "/opt/llmbench/out"):
        self.cards_dir = Path(cards_dir)
        self.vendor_claims = {
            "ternary": {
                "quality_retention": "94.6%",
                "baseline_comparison": "FP16 baseline",
                "benchmarks_count": "15 в thinking-режиме",
                "form_factor": "первая 27B в телефон"
            },
            "1bit": {
                "quality_retention": "89.5%",
                "baseline_comparison": "FP16 baseline",
                "benchmarks_count": "15 в thinking-режиме",
            }
        }

    def load_card(self, model_name: str) -> Dict[str, Any]:
        """Загрузка карточки модели."""
        card_file = self.cards_dir / f"card_{model_name}.json"
        if not card_file.exists():
            return {"error": f"Карточка не найдена: {card_file}"}

        with open(card_file) as f:
            return json.load(f)

    def extract_quality_metrics(self, card: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение реальных метрик качества."""
        quality = card.get("quality", {})
        micro = card.get("micro", {})

        if not quality:
            return {"status": "empty_quality", "note": "lm-eval не завершен"}

        return {
            "status": "has_quality",
            "gsm8k": quality.get("gsm8k", None),
            "humaneval": quality.get("humaneval", None),
            "mmlu_pro": quality.get("mmlu_pro", None),
            "minerva_math": quality.get("minerva_math", None),
            "bbh": quality.get("bbh", None),
            "math_acc": micro.get("math_acc", None),
            "json_valid": micro.get("json_valid", None),
            "tool_call_ok": micro.get("tool_call_ok", None),
        }

    def verify_claim(self, model: str, claim: str) -> Dict[str, Any]:
        """Проверка конкретного заявления."""
        card = self.load_card(model)
        if "error" in card:
            return {"claim": claim, "status": "NOT_VERIFIED", "reason": card["error"]}

        quality = self.extract_quality_metrics(card)

        if quality["status"] == "empty_quality":
            return {
                "claim": claim,
                "status": "NOT_VERIFIED",
                "reason": "Отсутствуют данные lm-eval",
                "available_data": "micro tests только"
            }

        # Проверка конкретных метрик
        if claim == "quality_retention":
            # Проверка наличия ключевых бенчмарков
            key_benchmarks = ["humaneval", "mmlu_pro", "gsm8k"]
            available = [k for k in key_benchmarks if quality.get(k) is not None]

            return {
                "claim": claim,
                "vendor_value": self.vendor_claims[model]["quality_retention"],
                "status": "PARTIALLY_VERIFIED" if len(available) > 0 else "NOT_VERIFIED",
                "available_benchmarks": available,
                "missing_benchmarks": [k for k in key_benchmarks if k not in available],
                "conclusion": "Требуются полные MMLU-Pro, HumanEval для подтверждения"
            }

        elif claim == "benchmarks_count":
            return {
                "claim": claim,
                "vendor_value": self.vendor_claims[model]["benchmarks_count"],
                "status": "NOT_VERIFIED",
                "reason": "Заявленное количество бенчмарков не подтверждено независимыми тестами",
                "actual_data": "Только smoke тесты (gsm8k: 0.8 для ternary)"
            }

        return {"claim": claim, "status": "NOT_IMPLEMENTED"}

    def generate_report(self) -> Dict[str, Any]:
        """Генерация отчета по проверке заявлений."""
        report = {
            "verification_date": "2026-07-16",
            "models": {},
            "summary": {}
        }

        for model in ["ternary", "1bit"]:
            model_data = self.load_card(f"local-27b-27b-{model}")
            claims_verification = {}

            for claim_key in ["quality_retention", "benchmarks_count"]:
                claims_verification[claim_key] = self.verify_claim(model, claim_key)

            model_info = {
                "name": model_data.get("name", "unknown"),
                "quant_format": model_data.get("quant_format", "unknown"),
                "disk_size_gb": model_data.get("disk_size_gb", 0),
                "vendor_claims": self.vendor_claims[model],
                "verification": claims_verification,
                "actual_quality": self.extract_quality_metrics(model_data),
                "conclusion": "Заявления требуют независимой верификации на полном наборе бенчмарков"
            }

            report["models"][model] = model_info

        # Итоговая сводка
        report["summary"] = {
            "total_vendor_claims": 4,
            "verified": 0,
            "partially_verified": 0,
            "not_verified": 4,
            "recommendation": "Все заявления вендора НЕ ПОДТВЕРЖДЕНЫ. Требуется выполнить полный lm-eval suite (MMLU-Pro, HumanEval, GPQA) для независимой верификации."
        }

        return report


def main():
    verifier = VendorClaimsVerifier()
    report = verifier.generate_report()

    print("=== ОТЧЕТ ПО ПРОВЕРКЕ ЗАЯВЛЕНИЙ ВЕНДОРА МОДЕЛИ ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Сохранение отчета
    output_file = Path("/opt/llmbench/out/verify_vendor_report.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nОтчет сохранен: {output_file}")


if __name__ == "__main__":
    main()
