"""
Валидация карточек моделей на консистентность данных.

Backward compatible: НЕ прерывает evaluate_model.py
Используется как post-processing для проверки качества данных.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ValidationResult:
    """Результат валидации"""
    valid: bool
    issues: List[str]
    warnings: List[str]
    card_path: str


class CardValidator:
    """Валидатор карточек моделей"""

    def __init__(self, thresholds_path: str = "config/thresholds.yaml"):
        self.thresholds_path = thresholds_path
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self) -> dict:
        """Загрузка порогов валидации"""
        try:
            import yaml
            with open(self.thresholds_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить thresholds: {e}")
            return {}

    def validate_card(self, card_path: str) -> ValidationResult:
        """Полная валидация карточки"""
        try:
            with open(card_path, "r", encoding="utf-8") as f:
                card = json.load(f)
        except Exception as e:
            return ValidationResult(
                valid=False,
                issues=[f"Ошибка чтения карточки: {e}"],
                warnings=[],
                card_path=card_path
            )

        issues = []
        warnings = []

        # 1. VRAM integrity (карточки пишут peak_vram_mb)
        hw = card.get("hardware", {})
        if hw:
            peak_vram = hw.get("peak_vram_mb", hw.get("peak_vram_mib", 0)) or 0
            disk_size = card.get("disk_size_gb", 0) * 1024

            if peak_vram < disk_size * 0.7:
                issues.append(
                    f"VRAM integrity FAIL: peak_vram={peak_vram} MiB < "
                    f"{disk_size * 0.7:.0f} MiB (70% disk)"
                )

        # 2. Quality не пустой
        quality = card.get("quality", {})
        if not quality:
            issues.append(
                "quality пустой - Gate-2 нельзя выносить на приёмку"
            )
        else:
            for task, score in quality.items():
                if not isinstance(score, (int, float)):
                    issues.append(f"quality.{task} не число: {type(score).__name__}")
                    continue
                if not (0 <= score <= 1):
                    issues.append(
                        f"quality.{task}={score} вне диапазона [0, 1]"
                    )
            meta = card.get("quality_meta") or {}
            if meta.get("partial"):
                warnings.append(
                    "quality частичный (quality_meta.partial=true) — Gate-2 рано"
                )

        # 3. Perf метрики
        perf = card.get("perf", {})
        if perf:
            tok_s = perf.get("gen_tok_s_mean", 0)
            if tok_s < 1.0:
                warnings.append(
                    f"Токены/секунда < 1 (tok_s={tok_s:.2f}) - возможно ошибка измерения"
                )

            ttft = perf.get("ttft_s_mean", 0)
            if ttft > 10:
                warnings.append(
                    f"TTFT > 10s (ttft={ttft:.2f}s) - очень медленный старт"
                )

        # 4. Cost метрики
        cost = card.get("cost", {})
        if cost:
            usd_per_1m = cost.get("usd_per_1m_tokens", 0)
            if usd_per_1m <= 0:
                warnings.append(
                    f"Стоимость не рассчитана (usd_per_1m_tokens={usd_per_1m})"
                )

        # 5. Модель венда
        if not card.get("repo"):
            issues.append("Отсутствует repo - источник модели не указан")

        # 6. Thresholds (если есть)
        if self.thresholds.get("quality"):
            for task, threshold in self.thresholds["quality"].items():
                if task in quality:
                    if quality[task] < threshold:
                        warnings.append(
                            f"quality.{task}={quality[task]:.3f} < threshold={threshold}"
                        )

        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            card_path=card_path
        )

    def validate_all(self, out_dir: str = "out") -> Dict[str, ValidationResult]:
        """Валидация всех карточек в директории"""
        results = {}

        for card_path in Path(out_dir).glob("card_*.json"):
            model_key = card_path.stem.replace("card_", "")
            results[model_key] = self.validate_card(str(card_path))

        return results


def print_validation_summary(results: Dict[str, ValidationResult]):
    """Вывод сводки валидации"""
    valid_count = sum(1 for r in results.values() if r.valid)
    total_count = len(results)

    print(f"\nВалидация карточек: {valid_count}/{total_count} валидных\n")

    for model_key, result in results.items():
        status = "✓ ВАЛИДНА" if result.valid else "✗ НЕВАЛИДНА"
        print(f"{status}: {model_key}")

        if result.issues:
            print(f"  Ошибки:")
            for issue in result.issues:
                print(f"    - {issue}")

        if result.warnings:
            print(f"  Предупреждения:")
            for warning in result.warnings:
                print(f"    - {warning}")

        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Валидация карточек моделей")
    parser.add_argument("--card", type=str, help="Путь к карточке")
    parser.add_argument("--all", action="store_true", help="Валидация всех карточек")
    parser.add_argument("--out-dir", default="out", help="Директория с карточками")
    parser.add_argument("--thresholds", default="config/thresholds.yaml", help="Путь к thresholds")

    args = parser.parse_args()

    validator = CardValidator(args.thresholds)

    if args.card:
        result = validator.validate_card(args.card)
        print(f"Карточка: {result.card_path}")
        print(f"Статус: {'ВАЛИДНА' if result.valid else 'НЕВАЛИДНА'}")
        if result.issues:
            print("Ошибки:")
            for issue in result.issues:
                print(f"  - {issue}")
        if result.warnings:
            print("Предупреждения:")
            for warning in result.warnings:
                print(f"  - {warning}")

    elif args.all:
        results = validator.validate_all(args.out_dir)
        print_validation_summary(results)

    else:
        parser.print_help()
