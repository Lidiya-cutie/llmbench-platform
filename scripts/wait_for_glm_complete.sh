#!/bin/bash
# Скрипт ожидания завершения glm-4-9b-chat прогона с автозапуском следующих шагов

LOG_FILE="/opt/llmbench/logs/eval_glm-4-9b-chat.log"
CARD_FILE="/opt/llmbench/out/card_glm-4-9b-chat.json"
CHECK_INTERVAL=300  # 5 минут

log_msg() {
    echo "[$(date -Iseconds)] $*" | tee -a /opt/llmbench/logs/glm_waiter.log
}

wait_for_completion() {
    local checks=0
    log_msg "Начало мониторинга завершения glm-4-9b-chat прогона"

    while true; do
        checks=$((checks + 1))

        # 1. Проверка процессов
        if ! pgrep -f "lm_eval.*glm-4-9b-chat" > /dev/null; then
            log_msg "lm-eval процесс завершился"

            # Даем немного времени на финализацию
            sleep 30

            # 2. Проверка успешности завершения
            if [ -f "$LOG_FILE" ]; then
                if tail -20 "$LOG_FILE" | grep -q "ERROR\|FATAL\|CalledProcessError"; then
                    log_msg "❌ Прогон завершился с ошибками! Проверьте лог: $LOG_FILE"
                    return 1
                fi
            fi

            # 3. Проверка обновления карточки
            if [ -f "$CARD_FILE" ]; then
                CARD_TIME=$(stat -c %Y "$CARD_FILE")
                NOW=$(date +%s)
                # Если карточка обновлена в последние 10 минут
                if [ $((NOW - CARD_TIME)) -lt 600 ]; then
                    log_msg "✅ Карточка обновлена! Прогон завершен успешно."

                    # Проверка quality поля
                    QUALITY=$(grep -A 2 '"quality"' "$CARD_FILE" | grep -c "^\s*}")
                    if [ "$QUALITY" -eq 0 ]; then
                        log_msg "⚠️  Карточка обновлена, но quality пустое - возможны проблемы"
                    else
                        log_msg "✅ Quality поле заполнено - полная успешность!"
                    fi

                    return 0
                else
                    log_msg "⚠️  Карточка не обновлялась давно - возможны проблемы"
                    return 1
                fi
            else
                log_msg "❌ Карточка не найдена: $CARD_FILE"
                return 1
            fi
        fi

        # Прогресс проверка
        if [ -f "$LOG_FILE" ]; then
            LATEST=$(tail -1 "$LOG_FILE" 2>/dev/null | grep -oP "Requesting API:.*?\| \d+/25026" | tail -1)
            if [ -n "$LATEST" ]; then
                CURRENT=$(echo "$LATEST" | grep -oP "\d+/25026" | grep -oP "\d+")
                if [ -n "$CURRENT" ]; then
                    PERCENT=$(echo "scale=2; $CURRENT * 100 / 25026" | bc)
                    log_msg "Прогресс: $CURRENT/25026 ($PERCENT%) - проверка #$checks"
                fi
            fi
        fi

        log_msg "Ожидание $CHECK_INTERVAL секунд... (проверка #$checks)"
        sleep $CHECK_INTERVAL
    done
}

# Запуск ожидания
if wait_for_completion; then
    log_msg "=== glm-4-9b-chat ЗАВЕРШЕН УСПЕШНО ==="
    log_msg "Теперь можно запускать следующие шаги:"
    log_msg "1. Остановить vLLM на порту 8077"
    log_msg "2. Запустить local-27b ternary на GPU1 порту 8078"
    log_msg "3. Запустить evaluate_model.py для local-27b-27b-ternary"
    log_msg "4. Повторить для 1-bit варианта"
    log_msg "5. Запустить исправленный Ollama"
    log_msg "6. Завершить compare_models"

    # Можно добавить автоматический запуск следующих шагов
    # /opt/llmbench/scripts/run_next_phase.sh
else
    log_msg "=== glm-4-9b-chat ЗАВЕРШИЛСЯ С ОШИБКАМИ ==="
    log_msg "Проверьте логи и вручную перезапустите прогон"
    exit 1
fi
