#!/bin/bash
# Интеграция code benchmarks в существующую систему /opt/llmbench
# Запуск EvalPlus для всех моделей без прерывания текущих процессов

set -euo pipefail

LLMBENCH=/opt/llmbench
VENV_PY="$LLMBENCH/.venv/bin/python"
SCRIPTS="$LLMBENCH/scripts"
OUT="$LLMBENCH/out"
LOGS="$LLMBENCH/logs"

echo "=== CODE BENCHMARKS INTEGRATION ==="
echo "Date: $(date -Iseconds)"
echo ""

# Проверяем что текущие бенчмарки работают
RUNNING_BENCHMARKS=$(ps aux | grep "[l]m_eval" | wc -l)
echo "Running lm_eval processes: $RUNNING_BENCHMARKS"
echo ""

# Модели для code benchmarking
MODELS=(
    "qwen3-8b-bf16"
    "ollama-qwen3-8b"
    "local-27b-27b-ternary"
    "deepseek-v4-flash"
    "gemma-3-27b-awq"
)

BENCHMARKS=(
    "humaneval+"
    "mbpp+"
)

# Проверяем готовность API endpoints
check_api_ready() {
    local model_key=$1
    local base_url=$2
    local max_retries=3
    local retry=0

    echo "Checking API for $model_key: $base_url"

    while [ $retry -lt $max_retries ]; do
        if curl -sf -m 5 "$base_url/models" >/dev/null 2>&1 || \
           curl -sf -m 5 "$base_url/v1/models" >/dev/null 2>&1; then
            echo "✅ API ready for $model_key"
            return 0
        fi
        retry=$((retry + 1))
        echo "⏳ API not ready (attempt $retry/$max_retries)"
        sleep 5
    done

    echo "❌ API not ready for $model_key"
    return 1
}

# Проверяем состояние моделей
echo "=== MODEL API STATUS ==="
declare -A MODEL_PORTS=(
    ["qwen3-8b-bf16"]="8077"
    ["ollama-qwen3-8b"]="11434"
    ["local-27b-27b-ternary"]="8082"
    ["deepseek-v4-flash"]="31.28.10.135:8000"
    ["gemma-3-27b-awq"]="19892"
)

for model in "${MODELS[@]}"; do
    port="${MODEL_PORTS[$model]}"
    if [[ "$port" == *":"* ]]; then
        # remote API
        check_api_ready "$model" "http://$port/v1"
    else
        check_api_ready "$model" "http://127.0.0.1:$port/v1"
    fi
done

echo ""
echo "=== CODE BENCHMARK PLAN ==="
echo "Models to benchmark: ${#MODELS[@]}"
echo "Benchmarks per model: ${#BENCHMARKS[@]}"
echo "Total runs: $((${#MODELS[@]} * ${#BENCHMARKS[@]}))"
echo ""

# Запуск code benchmarks
launch_code_benchmark() {
    local model=$1
    local benchmark=$2

    echo "🚀 Launching $benchmark for $model"

    local log_file="$LOGS/codebench_${model}_${benchmark}_$(date +%Y%m%d_%H%M%S).log"
    local output_dir="$OUT/codebench/$model/$benchmark"

    mkdir -p "$output_dir"

    # Запускаем в фоне
    nohup "$VENV_PY" "$SCRIPTS/run_evalplus_codebench.py" \
        --model "$model" \
        --benchmark "$benchmark" \
        --llmbench "$LLMBENCH" \
        --update-card \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "   PID: $pid"
    echo "   Log: $log_file"
    echo "   Output: $output_dir"
}

# Запускаем code benchmarks пачками (не более 2 одновременно)
MAX_PARALLEL=2
running=0

for model in "${MODELS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        # Проверяем уже выполненные
        if [ -f "$OUT/codebench/$model/$benchmark/eval_results.json" ]; then
            echo "⏭️  SKIP: $model/$benchmark (already completed)"
            continue
        fi

        # Ждем пока свободен слот
        while [ $running -ge $MAX_PARALLEL ]; do
            sleep 10
            running=$(ps aux | grep "[r]un_evalplus_codebench.py" | wc -l)
        done

        launch_code_benchmark "$model" "$benchmark"
        running=$((running + 1))
        sleep 2  # Небольшая пауза между запусками
    done
done

echo ""
echo "=== CODE BENCHMARKS LAUNCHED ==="
echo "Monitoring progress:"
echo "  tail -f $LOGS/codebench_*.log"
echo ""
echo "Results will be updated in model cards:"
echo "  $OUT/card_*.json"
echo ""
echo "🎯 Integration complete!"
