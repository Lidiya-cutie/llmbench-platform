#!/bin/bash
# Мониторинг прогресса glm-4-9b-chat lm-eval прогона

LOG_FILE="/opt/llmbench/logs/eval_glm-4-9b-chat.log"
CARD_FILE="/opt/llmbench/out/card_glm-4-9b-chat.json"

echo "=== Мониторинг glm-4-9b-chat lm-eval прогона ==="
echo "Время: $(date)"
echo

# 1. Проверка процессов
if pgrep -f "lm_eval.*glm-4-9b-chat" > /dev/null; then
    echo "✅ lm-eval进程运行中"
    PIDS=$(pgrep -f "lm_eval.*glm-4-9b-chat")
    echo "   PID: $PIDS"
else
    echo "❌ lm-eval进程未运行 - прогон завершился или упал"
fi

# 2. Проверка прогресса из лога
if [ -f "$LOG_FILE" ]; then
    LATEST=$(tail -1 "$LOG_FILE" 2>/dev/null | grep -oP "\d+/25026" | tail -1)
    if [ -n "$LATEST" ]; then
        CURRENT=$(echo "$LATEST" | grep -oP "^\d+")
        if [ -n "$CURRENT" ]; then
            PERCENT=$(echo "scale=2; $CURRENT * 100 / 25026" | bc)
            echo "📊 Прогресс: $CURRENT/25026 ($PERCENT%)"

            # Оценка оставшегося времени
            if [ "$CURRENT" -gt 10 ]; then
                LOG_START=$(stat -c %Y "$LOG_FILE")
                NOW=$(date +%s)
                ELAPSED=$((NOW - LOG_START))
                SEC_PER_REQ=$((ELAPSED / CURRENT))
                REMAINING=$((SEC_PER_REQ * (25026 - CURRENT)))
                HOURS=$((REMAINING / 3600))
                echo "⏱️  Оставшееся время: ~${HOURS} часов"
            fi
        fi
    else
        echo "⚠️  Прогресс не определен (прогон может завершиться)"
    fi
else
    echo "❌ Лог файл не найден: $LOG_FILE"
fi

# 3. Проверка обновления карточки
if [ -f "$CARD_FILE" ]; then
    CARD_TIME=$(stat -c %Y "$CARD_FILE")
    CARD_DATE=$(date -d "@$CARD_TIME" "+%Y-%m-%d %H:%M:%S")
    echo "📄 Карточка обновлена: $CARD_DATE"

    # Проверка quality поля
    QUALITY=$(grep -A 2 '"quality"' "$CARD_FILE" | grep -c "^\s*}")
    if [ "$QUALITY" -eq 0 ]; then
        echo "⚠️  Quality поле пустое - прогон еще не завершен"
    else
        echo "✅ Quality поле заполнено - прогон завершен успешно!"
    fi
else
    echo "❌ Карточка не найдена: $CARD_FILE"
fi

# 4. Проверка GPU
echo
echo "🔥 GPU статус:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader

echo
echo "=== Следующая проверка через 5 минут или наблюдайте лог: ==="
echo "   tail -f $LOG_FILE"
