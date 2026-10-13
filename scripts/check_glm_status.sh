#!/bin/bash
# Быстрая проверка статуса glm-4-9b-chat прогона

echo "=== glm-4-9b-chat Статус ==="

# 1. Процесс
if pgrep -f "lm_eval.*glm-4-9b-chat" > /dev/null; then
    echo "✅ Прогресс: АКТИВЕН"
    tail -1 /opt/llm-data/llmbench/logs/eval_glm-4-9b-chat.log 2>/dev/null | grep -oP "\d+/25026" | tail -1
else
    echo "⏸️  Прогресс: ЗАВЕРШЕН"
    ls -la /opt/llm-data/llmbench/out/card_glm-4-9b-chat.json 2>/dev/null | awk '{print "Карточка:", $6, $7, $8}'
fi

# 2. GPU
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | awk 'BEGIN{OFS=" | "} NR==1{print "GPU0:", $2, $3, "/", $4} NR==2{print "GPU1:", $2, $3, "/", $4}'

# 3. Прогноз (если активен)
if pgrep -f "lm_eval.*glm-4-9b-chat" > /dev/null; then
    LOG_FILE="/opt/llm-data/llmbench/logs/eval_glm-4-9b-chat.log"
    if [ -f "$LOG_FILE" ]; then
        CURRENT=$(tail -1 "$LOG_FILE" 2>/dev/null | grep -oP "^\s*\K\d+(?=25026)" | tail -1)
        if [ -n "$CURRENT" ] && [ "$CURRENT" -gt 10 ]; then
            LOG_START=$(stat -c %Y "$LOG_FILE")
            NOW=$(date +%s)
            ELAPSED=$((NOW - LOG_START))
            SEC_PER_REQ=$((ELAPSED / CURRENT))
            REMAINING_REQ=$((25026 - CURRENT))
            REMAINING_SEC=$((SEC_PER_REQ * REMAINING_REQ))
            HOURS=$((REMAINING_SEC / 3600))
            echo "⏱️  Осталось: ~${HOURS} часов"
        fi
    fi
fi
