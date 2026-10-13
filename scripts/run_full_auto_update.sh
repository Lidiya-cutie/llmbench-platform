#!/bin/bash
# Мастер-скрипт полной автоматизации обновления карточек
# Запускает все компоненты в правильном порядке

set -e

cd /opt/llmbench

echo "🚀 ПОЛНАЯ АВТОМАТИЗАЦИЯ ОБНОВЛЕНИЯ КАРТОЧЕК"
echo "=================================================="
echo ""

# 1. Проверка системы
echo "📊 1/4 Проверка доступности моделей..."
.venv/bin/python scripts/system_monitor_v2.py --json > /dev/null 2>&1
echo "   ✅ Системный мониторинг работает"
echo ""

# 2. Обновление всех карточек (merge + md генерация)
echo "📝 2/4 Интегрированное обновление карточек..."
.venv/bin/python scripts/auto_update_all_cards.py --out-dir out
echo ""

# 3. Проверка результатов
echo "📋 3/4 Проверка результатов..."
python3 -c "
from pathlib import Path
import json

out = Path('out')
md_files = list(out.glob('card_*.md'))
json_files = list(out.glob('card_*.json'))

print(f'   Markdown карточек: {len(md_files)}')
print(f'   JSON карточек: {len(json_files)}')

# Проверка свежести
import datetime
now = datetime.datetime.now()
recent_md = [f for f in md_files if datetime.datetime.fromtimestamp(f.stat().st_mtime) > now - datetime.timedelta(minutes=5)]
print(f'   Обновлено за последние 5 мин: {len(recent_md)} MD файлов')
"
echo ""

# 4. Создание отчета для презентации
echo "📊 4/4 Генерация презентационного отчета..."
.venv/bin/python scripts/system_monitor_v2.py --html > /dev/null 2>&1
echo "   ✅ System status dashboard обновлен"
echo ""

echo "=================================================="
echo "✅ ПОЛНАЯ АВТОМАТИЗАЦИЯ ЗАВЕРШЕНА"
echo ""
echo "📁 Результаты:"
echo "   - JSON карточки: out/card_*.json"
echo "   - Markdown карточки: out/card_*.md"
echo "   - Dashboard: out/dashboard_v2.html"
echo "   - System status: out/system_status_v2.html"
echo ""
echo "🔄 Для автоматического запуска добавьте в cron:"
echo "   */30 * * * * cd /opt/llmbench && bash scripts/run_full_auto_update.sh >> logs/auto_update.log 2>&1"