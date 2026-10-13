#!/usr/bin/env python3
"""
Автоматическая генерация Markdown карточек для всех моделей
Создает презентабельные отчеты из существующих JSON карточек
"""

from pathlib import Path
import json
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from llmbench import ModelCard, render_markdown, finalize

def load_card_from_json(json_path: Path) -> dict:
    """Загрузка карточки из JSON файла"""
    with open(json_path) as f:
        return json.load(f)

def convert_json_to_card(json_data: dict, model_key: str) -> ModelCard:
    """Конвертация JSON в ModelCard dataclass"""

    # Загружаем конфиг для получения метаданных модели
    config_path = Path("config/models.yaml")
    model_config = {}

    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
                model_config = cfg.get("models", {}).get(model_key, {})
        except:
            pass

    return ModelCard(
        name=model_key,
        repo=model_config.get("repo", json_data.get("repo", "")),
        base_model=model_config.get("base_model", json_data.get("base_model", "")),
        quant_format=model_config.get("quant_format", json_data.get("quant_format", "")),
        disk_size_gb=model_config.get("disk_size_gb", json_data.get("disk_size_gb", 0.0)),
        context_max=model_config.get("context_max", json_data.get("context_max", 0)),
        modalities=model_config.get("modalities", json_data.get("modalities", "text")),
        license=model_config.get("license", json_data.get("license", "")),
        engine=model_config.get("engine", json_data.get("engine", "")),
        serving_note=model_config.get("launch_cmd", json_data.get("serving_note", "")),
        thinking_mode=model_config.get("thinking_default", json_data.get("thinking_mode", "n/a")),

        # Данные из JSON карточки
        quality=json_data.get("quality", {}),
        micro=json_data.get("micro", {}),
        perf=json_data.get("perf", {}),
        context_scaling=json_data.get("context_scaling", []),
        hardware=json_data.get("hardware", {}),
        cost=json_data.get("cost", {}),
        reliability=json_data.get("reliability", {}),

        # Метаданные
        vendor_claims=json_data.get("vendor_claims", []),
        conclusion=json_data.get("conclusion", ""),
        fit=json_data.get("fit", {})
    )

def main():
    out_dir = Path("out")

    if not out_dir.exists():
        print("❌ Директория out/ не найдена")
        return

    print("🚀 Автоматическая генерация Markdown карточек")
    print("=" * 60)

    # Находим все JSON карточки
    json_cards = sorted(out_dir.glob("card_*.json"))

    if not json_cards:
        print("❌ JSON карточки не найдены")
        return

    generated = 0
    failed = 0

    for json_path in json_cards:
        # Пропускаем backup файлы
        if ".bak." in json_path.name:
            continue

        model_key = json_path.stem.replace("card_", "")
        md_path = out_dir / f"{json_path.stem}.md"

        print(f"\n📝 Обработка {model_key}...")

        try:
            # Загружаем JSON
            json_data = load_card_from_json(json_path)

            # Конвертируем в ModelCard
            card = convert_json_to_card(json_data, model_key)

            # Генерируем заключение если его нет
            if not card.conclusion:
                print(f"   → Генерация заключения...")
                card = finalize(card)

            # Рендерим Markdown
            markdown = render_markdown(card)

            # Сохраняем
            md_path.write_text(markdown, encoding="utf-8")

            print(f"   ✅ Создан: {md_path.name}")
            generated += 1

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 ИТОГО: {generated} создано, {failed} ошибок")
    print(f"📁 Карточки сохранены в {out_dir}/")

    if generated > 0:
        print(f"\n💡 Для просмотра откройте файл в браузере или Markdown viewer")
        print(f"   Пример: file://{out_dir.absolute()}/card_qwen2.5-vl-7b.md")

if __name__ == "__main__":
    main()