#!/bin/bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-out}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"

echo "Мониторинг состояния GPU, процессов и quality"
echo "Интервал: $CHECK_INTERVAL секунд"
echo "Директория: $OUT_DIR"
echo "Нажмите Ctrl+C для остановки"

while true; do
    clear
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo

    # Состояние GPU
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
        awk -F',' '{printf "GPU%s: %s - %s/%s MiB (%s загрузка)\n", $1, $2, $3, $4, $5}'
    echo

    # Активные процессы оценки
    echo "--- Активные процессы lm-eval ---"
    ps aux | grep -E 'evaluate_model|lm-eval' | grep -v grep | \
        awk '{printf "PID %s: %s (время: %s)\n", $2, $11, $10}'
    echo

    # Статус качества
    echo "--- Quality статус ---"
    python /opt/llmbench/scripts/monitor_quality.py \
        --out-dir "$OUT_DIR" \
        --oneshot 2>/dev/null | grep -E '(Quality готов|✓|✗)'
    echo

    # Docker контейнеры
    echo "--- Docker контейнеры ---"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | \
        grep -E '(vllm|NAMES|local-27b)' || echo "Нет активных контейнеров"
    echo

    # Свободное место
    echo "--- Диск ---"
    df -h /opt | awk 'NR==2 {printf "Доступно: %s из %s (%s использовано)\n", $4, $2, $5}'
    echo

    sleep "$CHECK_INTERVAL"
done
