#!/bin/bash
set -euo pipefail

# Автоматическое сравнение моделей с определением baseline по quality
# Запускается после завершения lm-eval минимум для одной модели

OUT_DIR="${OUT_DIR:-out}"
MIN_QUALITY_MODELS="${MIN_QUALITY_MODELS:-1}"
BASELINE_MODEL="${BASELINE_MODEL:-auto}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-scripts/compare_models.py}"

echo "Автоматическое сравнение моделей"
echo "Директория: $OUT_DIR"
echo "Минимум моделей с quality: $MIN_QUALITY_MODELS"
echo "Baseline: $BASELINE_MODEL"

# Ожидание quality
python scripts/monitor_quality.py \
    --min-models "$MIN_QUALITY_MODELS" \
    --interval 300 \
    --out-dir "$OUT_DIR" | tee /tmp/quality_monitor.log

# Определение baseline
if [[ "$BASELINE_MODEL" == "auto" ]]; then
    BASELINE_MODEL=$(jq -r 'select(.quality | length > 0) | .model_key' \
        "$OUT_DIR"/card_*.json | head -1)
    echo "Автоматический baseline: $BASELINE_MODEL"
fi

# Сбор всех моделей с quality
MODELS_WITH_QUALITY=$(jq -r 'select(.quality | length > 0) | .model_key' \
    "$OUT_DIR"/card_*.json | tr '\n' ',' | sed 's/,$//')

echo "Запуск сравнения: baseline=$BASELINE_MODEL, models=$MODELS_WITH_QUALITY"

python "$COMPARE_SCRIPT" \
    --baseline "$BASELINE_MODEL" \
    --models "$MODELS_WITH_QUALITY" \
    --weights quality:0.4,perf:0.3,cost:0.3 \
    --out-dir "$OUT_DIR/compare_runs"

echo "Сравнение завершено: $OUT_DIR/compare_runs/compare_*.md"
