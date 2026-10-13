#!/usr/bin/env python3
"""
Интегрированная система автоматического обновления карточек
Объединяет merge_lmeval_into_card + auto_generate_cards для полной автоматизации
"""

from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class CardUpdateManager:
    """Менеджер обновления карточек с автоматической md генерацией"""

    def __init__(self, out_dir="out"):
        self.out_dir = Path(out_dir)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'merged': [],
            'generated': [],
            'skipped': [],
            'failed': []
        }

    def find_models_with_new_data(self) -> list[dict]:
        """Найти модели, которые имеют новые lm_eval данные но не влиты в карточку"""
        models_info = []

        for card_file in sorted(self.out_dir.glob("card_*.json")):
            if ".bak." in card_file.name:
                continue

            try:
                card = json.loads(card_file.read_text())
                model_key = card.get("name", card_file.stem.replace("card_", ""))

                # Проверяем есть ли lm_eval данные для этой модели
                lmeval_dirs = [
                    self.out_dir / f"lmeval_{model_key}_queue",
                    self.out_dir / f"lmeval_{model_key}",
                    self.out_dir / f"lmeval_{model_key.replace('-8b-bf16', '-8b-bf16_queue')}"
                ]

                has_new_data = False
                lmeval_root = None

                for lmeval_dir in lmeval_dirs:
                    if lmeval_dir.exists() and any(lmeval_dir.iterdir()):
                        has_new_data = True
                        lmeval_root = lmeval_dir
                        break

                # Проверяем пропущенные метрики
                quality = card.get("quality", {})
                quality_meta = card.get("quality_meta", {})
                sources = quality_meta.get("sources", {})

                # Если есть lm_eval директория но нет_sources или есть ошибки
                needs_update = False
                update_reason = ""

                if has_new_data and not sources:
                    needs_update = True
                    update_reason = "new_lm_eval_data"
                elif any(sources.get(task, {}).get("error") for task in ["mmlu_pro", "humaneval", "gsm8k"] if task in sources):
                    needs_update = True
                    update_reason = "fix_merge_errors"
                elif quality.get("humaneval", "MISSING") == "MISSING" and has_new_data:
                    needs_update = True
                    update_reason = "add_humaneval"

                if needs_update:
                    models_info.append({
                        'model_key': model_key,
                        'card_file': card_file,
                        'lmeval_root': lmeval_root,
                        'reason': update_reason
                    })

            except Exception as e:
                print(f"⚠️ Ошибка проверки {card_file.name}: {e}")

        return models_info

    def merge_card(self, model_info: dict) -> bool:
        """Выполнить merge для модели"""
        model_key = model_info['model_key']
        card_file = model_info['card_file']
        lmeval_root = model_info['lmeval_root']

        print(f"   📝 Merge {model_key}...")

        try:
            # Запускаем merge скрипт
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/merge_lmeval_into_card.py",
                    "--card", str(card_file),
                    "--lmeval-root", str(lmeval_root)
                ],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print(f"      ✅ Merge успешен")
                self.results['merged'].append(model_key)
                return True
            else:
                print(f"      ❌ Merge failed: {result.stderr[:100]}")
                self.results['failed'].append({'model': model_key, 'error': result.stderr[:200]})
                return False

        except Exception as e:
            print(f"      ❌ Merge error: {e}")
            self.results['failed'].append({'model': model_key, 'error': str(e)})
            return False

    def generate_markdown(self, model_key: str) -> bool:
        """Генерировать markdown карточку"""
        print(f"   📄 Генерация MD для {model_key}...")

        try:
            # Импортируем функции генерации
            from llmbench import ModelCard, render_markdown, finalize

            # Загружаем обновленный JSON
            card_file = self.out_dir / f"card_{model_key}.json"
            json_data = json.loads(card_file.read_text())

            # Конвертируем в ModelCard
            card = ModelCard(
                name=model_key,
                repo=json_data.get("repo", ""),
                base_model=json_data.get("base_model", ""),
                quant_format=json_data.get("quant_format", ""),
                disk_size_gb=json_data.get("disk_size_gb", 0.0),
                context_max=json_data.get("context_max", 0),
                modalities=json_data.get("modalities", "text"),
                license=json_data.get("license", ""),
                engine=json_data.get("engine", ""),
                serving_note=json_data.get("serving_note", ""),
                thinking_mode=json_data.get("thinking_mode", "n/a"),
                quality=json_data.get("quality", {}),
                micro=json_data.get("micro", {}),
                perf=json_data.get("perf", {}),
                context_scaling=json_data.get("context_scaling", []),
                hardware=json_data.get("hardware", {}),
                cost=json_data.get("cost", {}),
                reliability=json_data.get("reliability", {}),
                conclusion=json_data.get("conclusion", ""),
                fit=json_data.get("fit", {}),
                vendor_claims=json_data.get("vendor_claims", [])
            )

            # Генерируем заключение если нужно
            if not card.conclusion:
                card = finalize(card)

            # Рендерим markdown
            markdown = render_markdown(card)

            # Сохраняем
            md_path = self.out_dir / f"card_{model_key}.md"
            md_path.write_text(markdown, encoding="utf-8")

            print(f"      ✅ MD создан")
            self.results['generated'].append(model_key)
            return True

        except Exception as e:
            print(f"      ❌ MD generation error: {e}")
            self.results['failed'].append({'model': model_key, 'error': f"MD: {str(e)}"})
            return False

    def update_dashboard(self) -> bool:
        """Обновить dashboard v2"""
        print("   📊 Обновление dashboard...")

        try:
            result = subprocess.run(
                [sys.executable, "scripts/generate_dashboard_v2.py"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print("      ✅ Dashboard обновлен")
                return True
            else:
                print(f"      ⚠️ Dashboard warning: {result.stderr[:100]}")
                return False

        except Exception as e:
            print(f"      ❌ Dashboard error: {e}")
            return False

    def run_full_update(self):
        """Полное обновление всех карточек"""
        print("🚀 ИНТЕГРИРОВАННОЕ ОБНОВЛЕНИЕ ВСЕХ КАРТОЧЕК")
        print("=" * 60)

        # 1. Находим модели с новыми данными
        models_to_update = self.find_models_with_new_data()

        if not models_to_update:
            print("✅ Все карточки актуальны, новых данных нет")
        else:
            print(f"📝 Найдено {len(models_to_update)} моделей для обновления")

            for model_info in models_to_update:
                model_key = model_info['model_key']
                reason = model_info['reason']
                print(f"\n🔄 {model_key} ({reason})")

                # Merge
                if self.merge_card(model_info):
                    # Генерация MD после успешного merge
                    self.generate_markdown(model_key)
                else:
                    print(f"      ⚠️ Пропуск MD из-за ошибок merge")

        # 2. Генерируем markdown для всех моделей (включая те, что не менялись)
        print(f"\n📄 Генерация Markdown для всех моделей...")
        all_updated = set(self.results['merged'] + self.results['generated'])

        for card_file in sorted(self.out_dir.glob("card_*.json")):
            if ".bak." in card_file.name:
                continue

            model_key = card_file.stem.replace("card_", "")
            if model_key not in all_updated:
                self.generate_markdown(model_key)

        # 3. Обновляем dashboard
        self.update_dashboard()

        # 4. Показываем итоги
        self.print_summary()

    def print_summary(self):
        """Вывод итогов обновления"""
        print("\n" + "=" * 60)
        print("📊 ИТОГИ ОБНОВЛЕНИЯ:")
        print(f"   ✅ Обновлено merge: {len(self.results['merged'])}")
        print(f"   📄 Создано MD: {len(self.results['generated'])}")
        print(f"   ⏭️ Пропущено: {len(self.results['skipped'])}")
        print(f"   ❌ Ошибок: {len(self.results['failed'])}")

        if self.results['merged']:
            print(f"\n🔄 Обновленные модели: {', '.join(self.results['merged'])}")

        if self.results['failed']:
            print(f"\n⚠️ Ошибки:")
            for fail in self.results['failed']:
                print(f"   - {fail['model']}: {fail.get('error', 'unknown')[:80]}")

        print(f"\n💾 Все карточки сохранены в {self.out_dir}/")
        print(f"📊 Dashboard: {self.out_dir}/dashboard_v2.html")
        print(f"🕐 Обновлено: {self.results['timestamp']}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out", help="Директория с карточками")
    args = ap.parse_args()

    manager = CardUpdateManager(out_dir=args.out_dir)
    manager.run_full_update()


if __name__ == "__main__":
    main()