#!/usr/bin/env python3
"""Диагностика причин нулевых результатов HumanEval для фикса.

Фаза 1 из чеклиста: анализ raw samples, проверка max_gen_toks, флагов lm-eval.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class HumanEvalDiagnostics:
    """Анализ проблем с HumanEval benchmark."""

    def __init__(self, llmbench_dir: Path):
        self.llmbench = llmbench_dir
        self.out_dir = llmbench_dir / "out"
        self.logs_dir = llmbench_dir / "logs"

    def find_humaneval_samples(self, model_key: str) -> List[Path]:
        """Поиск raw samples от lm-eval HumanEval."""

        # Стандартные пути lm-eval
        search_paths = [
            self.out_dir / f"lmeval_{model_key}_queue" / "humaneval",
            self.out_dir / "lmeval_humaneval" / model_key,
            self.out_dir / "humaneval" / model_key,
        ]

        samples_files = []
        for path in search_paths:
            if path.exists():
                # Ищем файлы с samples
                samples_files.extend(path.rglob("samples*.json"))
                samples_files.extend(path.rglob("*_samples.json"))

        return samples_files

    def analyze_raw_samples(self, samples_file: Path) -> Dict[str, Any]:
        """Анализ raw samples на предмет проблем."""

        try:
            data = json.loads(samples_file.read_text())
        except Exception as e:
            return {"error": f"Failed to load {samples_file}: {e}"}

        diagnostics = {
            "file": str(samples_file),
            "total_samples": len(data) if isinstance(data, list) else 0,
            "issues": [],
        }

        if not isinstance(data, list):
            diagnostics["error"] = f"Unexpected format: {type(data)}"
            return diagnostics

        # Анализ каждого sample
        truncated_count = 0
        no_code_count = 0
        has_thinking = 0
        avg_length = 0
        total_length = 0

        for i, sample in enumerate(data[:50]):  # Первые 50 для быстрого анализа
            if not isinstance(sample, dict):
                continue

            # Проверка на обрезку
            generated = sample.get("generated_response", "") or sample.get("generation", "")

            if not generated:
                no_code_count += 1
                diagnostics["issues"].append(f"Sample {i}: empty generation")
                continue

            total_length += len(generated)

            # Проверка на ```` блок
            if "````" in generated or len(generated) < 100:
                truncated_count += 1
                diagnostics["issues"].append(f"Sample {i}: likely truncated (```` or short)")

            # Проверка на thinking/reasoning блоки
            if "<thinking>" in generated or "<think" in generated:
                has_thinking += 1

            # Проверка на наличие кода
            if not re.search(r"def\s+\w+\s*\(", generated):
                no_code_count += 1
                diagnostics["issues"].append(f"Sample {i}: no function definition found")

        if data:
            avg_length = total_length / min(len(data), 50)

        diagnostics.update({
            "truncated_rate": truncated_count / min(len(data), 50) if data else 0,
            "no_code_rate": no_code_count / min(len(data), 50) if data else 0,
            "has_thinking": has_thinking > 0,
            "thinking_samples": has_thinking,
            "avg_generation_length": avg_length,
        })

        return diagnostics

    def check_lm_eval_flags(self, model_key: str) -> Dict[str, Any]:
        """Проверка флагов lm-eval из логов."""

        log_files = [
            self.logs_dir / f"lm_eval_{model_key}_queue.log",
            self.logs_dir / f"{model_key}_humaneval.log",
            self.logs_dir / "after_glm_queue.log",
        ]

        flags_check = {
            "has_confirm_run_unsafe_code": False,
            "has_apply_chat_template": False,
            "max_gen_toks": None,
        }

        for log_file in log_files:
            if not log_file.exists():
                continue

            log_content = log_file.read_text()

            if "--confirm_run_unsafe_code" in log_content:
                flags_check["has_confirm_run_unsafe_code"] = True

            if "--apply_chat_template" in log_content:
                flags_check["has_apply_chat_template"] = True

            # Попытка извлечь max_gen_toks
            match = re.search(r"--max_gen_toks\s+(\d+)", log_content)
            if match:
                flags_check["max_gen_toks"] = int(match.group(1))

        return flags_check

    def check_model_config(self, model_key: str) -> Dict[str, Any]:
        """Проверка конфигурации модели."""

        card_file = self.out_dir / f"card_{model_key}.json"

        if not card_file.exists():
            return {"error": f"Card not found: {card_file}"}

        card = json.loads(card_file.read_text())

        config = {
            "model": card.get("base_model", "unknown"),
            "engine": card.get("engine", "unknown"),
            "quantization": card.get("quant_format", "unknown"),
            "current_humaneval_score": card.get("quality", {}).get("humaneval", None),
        }

        # Проверка на reasoning capability
        thinking = card.get("thinking_mode", "")
        if thinking and "off" not in thinking.lower():
            config["is_reasoning"] = True

        return config

    def run_full_diagnosis(self, model_key: str) -> Dict[str, Any]:
        """Полная диагностика модели."""

        print(f"=== DIAGNOSING {model_key} ===")

        diagnosis = {
            "model": model_key,
            "timestamp": "2026-07-31T07:00:00Z",
            "sections": {},
        }

        # 1. Анализ raw samples
        print("1. Analyzing raw samples...")
        samples_files = self.find_humaneval_samples(model_key)

        if samples_files:
            samples_diag = self.analyze_raw_samples(samples_files[0])
            diagnosis["sections"]["samples_analysis"] = samples_diag
            print(f"   Found {len(samples_files)} sample files")
            print(f"   Truncated rate: {samples_diag.get('truncated_rate', 0):.1%}")
            print(f"   No code rate: {samples_diag.get('no_code_rate', 0):.1%}")
        else:
            diagnosis["sections"]["samples_analysis"] = {"error": "No sample files found"}
            print("   WARNING: No sample files found")

        # 2. Проверка флагов lm-eval
        print("2. Checking lm-eval flags...")
        flags_diag = self.check_lm_eval_flags(model_key)
        diagnosis["sections"]["lm_eval_flags"] = flags_diag
        print(f"   --confirm_run_unsafe_code: {flags_diag['has_confirm_run_unsafe_code']}")
        print(f"   --apply_chat_template: {flags_diag['has_apply_chat_template']}")
        print(f"   max_gen_toks: {flags_diag['max_gen_toks'] or 'default'}")

        # 3. Проверка конфигурации модели
        print("3. Checking model configuration...")
        config_diag = self.check_model_config(model_key)
        diagnosis["sections"]["model_config"] = config_diag
        print(f"   Current HumanEval score: {config_diag.get('current_humaneval_score')}")
        print(f"   Is reasoning: {config_diag.get('is_reasoning', False)}")

        # 4. Выводы и рекомендации
        print("4. RECOMMENDATIONS:")
        recommendations = self._generate_recommendations(diagnosis)
        diagnosis["recommendations"] = recommendations

        for rec in recommendations:
            print(f"   - {rec}")

        return diagnosis

    def _generate_recommendations(self, diagnosis: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по диагнозу."""

        recommendations = []

        samples = diagnosis["sections"].get("samples_analysis", {})
        flags = diagnosis["sections"].get("lm_eval_flags", {})
        config = diagnosis["sections"].get("model_config", {})

        # Проблема с max_gen_toks
        if not flags.get("max_gen_toks") or flags.get("max_gen_toks", 0) < 2048:
            recommendations.append(
                "INCREASE max_gen_toks to 2048-4096+ (current: %s)" % flags.get("max_gen_toks", "default")
            )

        # Проблема с unsafe code flag
        if not flags.get("has_confirm_run_unsafe_code"):
            recommendations.append(
                "ADD --confirm_run_unsafe_code flag for HumanEval execution"
            )

        # Проблема с chat template
        if not flags.get("has_apply_chat_template") and config.get("current_humaneval_score") == 0.0:
            recommendations.append(
                "ADD --apply_chat_template for chat/instruct models"
            )

        # Проблема с extraction
        if samples.get("truncated_rate", 0) > 0.3:
            recommendations.append(
                "FIX EXTRACTION: %.0f%% samples truncated (use EvalPlus sanitize)" % (samples.get("truncated_rate", 0) * 100)
            )

        # Проблема с thinking blocks
        if samples.get("has_thinking") and samples.get("thinking_samples", 0) > 5:
            recommendations.append(
                "ENABLE reasoning-parser for %d thinking samples (vLLM/Ollama)" % samples.get("thinking_samples", 0)
            )

        # Если все ок
        if not recommendations and config.get("current_humaneval_score", 0) > 0:
            recommendations.append("NO CRITICAL ISSUES: Current setup looks good")

        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Diagnose HumanEval failures")
    parser.add_argument("--model", required=True, help="Model key (e.g., qwen3-8b-bf16)")
    parser.add_argument("--llmbench", default="/opt/llmbench", type=Path)
    parser.add_argument("--output", type=Path, help="Save diagnosis to JSON file")

    args = parser.parse_args()

    diagnostics = HumanEvalDiagnostics(args.llmbench)
    diagnosis = diagnostics.run_full_diagnosis(args.model)

    if args.output:
        args.output.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False))
        print(f"\nDiagnosis saved to: {args.output}")

    # Exit code based on critical issues
    critical_issues = any(
        "INCREASE" in rec or "ADD" in rec or "FIX" in rec
        for rec in diagnosis.get("recommendations", [])
    )

    return 1 if critical_issues else 0


if __name__ == "__main__":
    exit(main())
