#!/usr/bin/env python3
"""Интеграция code метрик в существующую систему дашбордов.

Добавляет поддержку EvalPlus результатов (HumanEval+, MBPP+) в:
- generate_dashboard_v2.py
- compare_models.py
- Model cards (card_*.json)

Code метрики будут отображаться В ТОМ ЖЕ САМОМ дашборде рядом с quality/perf/hardware.
"""
from pathlib import Path
import json
import shutil

LLMBENCH = Path("/opt/llmbench")

def integrate_code_metrics_to_dashboard():
    """Добавление code секции в generate_dashboard_v2.py"""

    dashboard_file = LLMBENCH / "scripts" / "generate_dashboard_v2.py"

    if not dashboard_file.exists():
        print(f"ERROR: {dashboard_file} not found")
        return False

    content = dashboard_file.read_text()

    # Проверяем есть ли уже code секция
    if "CODE_COLS" in content or "code_metrics" in content:
        print("Code metrics уже интегрированы в dashboard")
        return True

    # Находим место для вставки CODE_COLS после MICRO_COLS
    micro_cols_pos = content.find("MICRO_COLS = (")
    if micro_cols_pos == -1:
        print("ERROR: MICRO_COLS not found in dashboard")
        return False

    # Добавляем CODE_COLS после MICRO_COLS
    code_cols_definition = """
CODE_METRICS_COLS = (
    ("humaneval_plus", "HumanEval+"),
    ("mbpp_plus", "MBPP+"),
)
"""

    # Находим конец MICRO_COLS definition
    micro_end = content.find(")\n", micro_cols_pos) + 2
    content = content[:micro_end] + code_cols_definition + content[micro_end:]

    # Добавляем загрузку code метрик в _normalize_card
    normalize_pos = content.find("def _normalize_card(self, card:")
    if normalize_pos == -1:
        print("ERROR: _normalize_card not found")
        return False

    # Находим место где извлекаются micro метрики
    micro_extract_pos = content.find("micro = card.get(\"micro\", {})", normalize_pos)
    if micro_extract_pos == -1:
        print("ERROR: micro extraction not found")
        return False

    # Добавляем извлечение code метрик после micro
    code_extract = """
        # Extract code metrics from EvalPlus results
        code = card.get("code", {})
        code_metrics = {k: norm(v) if isinstance(v, (int, float)) else v
                       for k, v in code.items()
                       if any(k.endswith(suffix) for suffix in ["_plus", "pass@1", "pass@10"])}
"""

    # Находим конец normalization секции и вставляем code извлечение
    norm_function_end = content.find("\n        return", normalize_pos)
    if norm_function_end == -1:
        print("ERROR: normalization end not found")
        return False

    # Находим строку перед return
    return_start = content.rfind("\n", 0, norm_function_end)
    content = content[:return_start+1] + code_extract + "\n" + content[return_start+1:]

    # Обновляем return statement чтобы включить code_metrics
    return_pos = content.find("return {", normalize_pos)
    if return_pos == -1:
        print("ERROR: return statement not found")
        return False

    # Находим конец return dict
    return_end = content.find("\n        }", return_pos)
    return_content = content[return_pos:return_end+2]

    # Добавляем code_metrics в return
    updated_return = return_content.replace(
        "\"micro\": micro,",
        "\"micro\": micro,\n        \"code_metrics\": code_metrics,"
    )

    content = content[:return_pos] + updated_return + content[return_end+2:]

    # Backup и сохранение
    backup_file = dashboard_file.with_suffix(".py.bak-code-integration")
    shutil.copy(dashboard_file, backup_file)

    dashboard_file.write_text(content)

    print(f"✅ Code metrics интегрированы в {dashboard_file}")
    print(f"📁 Backup: {backup_file}")
    return True

def integrate_code_metrics_to_compare():
    """Добавление code метрик в compare_models.py"""

    compare_file = LLMBENCH / "scripts" / "compare_models.py"

    if not compare_file.exists():
        print(f"ERROR: {compare_file} not found")
        return False

    content = compare_file.read_text()

    # Проверяем есть ли уже code метрики
    if "humaneval_plus" in content or "mbpp_plus" in content:
        print("Code metrics уже в compare_models.py")
        return True

    # Добавляем code метрики в QUALITY_COLS
    quality_cols_pos = content.find("QUALITY_COLS = (")
    if quality_cols_pos == -1:
        print("ERROR: QUALITY_COLS not found")
        return False

    # Находим конец QUALITY_COLS
    quality_end = content.find(")\n", quality_cols_pos)
    quality_content = content[quality_cols_pos:quality_end+2]

    # Добавляем code метрики
    updated_quality = quality_content.replace(
        '"humaneval",\n',
        '"humaneval",\n    "humaneval_plus",\n    "mbpp_plus",\n'
    )

    content = content[:quality_cols_pos] + updated_quality + content[quality_end+2:]

    # Backup и сохранение
    backup_file = compare_file.with_suffix(".py.bak-code-integration")
    shutil.copy(compare_file, backup_file)

    compare_file.write_text(content)

    print(f"✅ Code metrics добавлены в {compare_file}")
    print(f"📁 Backup: {backup_file}")
    return True

def create_code_integration_example():
    """Создать пример интеграции code результатов в model cards."""

    example_card = LLMBENCH / "out" / "card_example_with_code.json"

    example = {
        "name": "model-name-code-example",
        "base_model": "Model-Name-7B",
        "quant_format": "BF16",
        "quality": {
            "gsm8k": 0.75,
            "minerva_math": 0.60,
            "mmlu_pro": 0.68,
            "humaneval": 0.0,  # Старый lm-eval (будет заменен)
        },
        "code": {
            "humaneval_plus": {
                "pass@1": 0.82,  # EvalPlus результат
                "pass@10": 0.95,
                "samples": 164,
                "timestamp": "2026-07-31T07:30:00Z",
                "backend": "openai",
                "version": "evalplus-0.3.1"
            },
            "mbpp_plus": {
                "pass@1": 0.76,
                "pass@10": 0.92,
                "samples": 974,
                "timestamp": "2026-07-31T07:45:00Z",
                "backend": "openai",
                "version": "evalplus-0.3.1"
            }
        },
        "perf": {
            "gen_tok_s_mean": 45.2,
        },
        "hardware": {
            "peak_vram_mb": 8192,
        }
    }

    example_card.write_text(json.dumps(example, indent=2, ensure_ascii=False))
    print(f"📄 Пример карточки создан: {example_card}")

    return True

def main():
    """Главная функция интеграции."""

    print("=== ИНТЕГРАЦИЯ CODE МЕТРИК В ДАШБОРД ===")
    print()

    success = True

    # 1. Интеграция в generate_dashboard_v2.py
    print("1. Интеграция в generate_dashboard_v2.py...")
    if not integrate_code_metrics_to_dashboard():
        success = False

    # 2. Интеграция в compare_models.py
    print("\n2. Интеграция в compare_models.py...")
    if not integrate_code_metrics_to_compare():
        success = False

    # 3. Создание примера карточки
    print("\n3. Создание примера карточки...")
    if not create_code_integration_example():
        success = False

    if success:
        print("\n✅ CODE МЕТРИКИ УСПЕШНО ИНТЕГРИРОВАНЫ")
        print("\n📊 ЧТО БУДЕТ ОТОБРАЖАТЬСЯ В ДАШБОРДЕ:")
        print("   - HumanEval+ pass@1 (вместо старого humaneval: 0.0)")
        print("   - MBPP+ pass@1")
        print("   - Code секция в model cards")
        print("   - Интеграция с существующей таблицей сравнения")
        print("\n🔄 СЛЕДУЮЩИЙ ШАГ:")
        print("   python scripts/generate_dashboard_v2.py")
    else:
        print("\n❌ ОШИБКИ ИНТЕГРАЦИИ")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
