#!/bin/bash
# ИСПРАВЛЕННЫЙ ночной сравнительный бенчмарк: vLLM + Ollama + local-27b ternary
# ИСПРАВЛЕНИЯ: увеличен таймаут, упрощен suite, фиксы Ollama, пропуск сломанных моделей
# Лог: logs/nightly_fixed.log

set -euo pipefail

LLMBENCH=/opt/llm-data/llmbench
LOCAL27B=/opt/llm-data/local-27b-demo
MLOPS=/opt/mlops-stack
QWEN25VL_SNAP="hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
LOGDIR="$LLMBENCH/logs"
mkdir -p "$LOGDIR"
MAIN_LOG="$LOGDIR/nightly_fixed.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MAIN_LOG"; }

wait_openai() {
  local url="$1"
  local expect_model="${2:-}"
  local tries="${3:-180}"
  for i in $(seq 1 "$tries"); do
    if resp=$(curl -sf "${url%/}/models" 2>/dev/null); then
      if [ -z "$expect_model" ] || echo "$resp" | grep -q "$expect_model"; then
        log "API ready: $url model=${expect_model:-any} (${i}×5s)"
        return 0
      fi
    fi
    sleep 5
  done
  log "ERROR: API not ready: $url (expect=$expect_model)"
  return 1
}

free_port() {
  local port="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" 2>/dev/null || true
  fi
  sleep 2
}

stop_inference() {
  log "Stopping Triton / vLLM / bench-Ollama (сохраняем работающий local-27b ternary)..."
  (cd "$MLOPS" && docker compose --profile inference stop triton) 2>&1 | tee -a "$MAIN_LOG" || true
  docker rm -f vllm-bench 2>/dev/null || true
  # НЕ останавливаем local-27b ternary на GPU1 - он работает!
  stop_ollama_bench
  free_port 8077
  sleep 3
}

start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local extra_args="${3:-}"
  log "Starting vLLM on GPU0 :8077 — $served_name"
  docker rm -f vllm-bench 2>/dev/null || true
  free_port 8077
  # shellcheck disable=SC2086
  docker run -d --name vllm-bench \
    --gpus '"device=0"' \
    -v /opt/llm-data/.cache_huggingface:/root/.cache/huggingface \
    -p 8077:8077 \
    mlops-triton-vllm:24.12-qwen3 \
    vllm serve "$model_path" \
    --host 0.0.0.0 --port 8077 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --served-model-name "$served_name" \
    $extra_args
  wait_openai "http://127.0.0.1:8077/v1" "$served_name"
}

start_ollama_gpu1() {
  # Проверяем что local-27b ternary завершился
  if pgrep -f "llama-server.*Ternary-local-27b-Q2_0.gguf" > /dev/null; then
    log "local-27b ternary еще работает на GPU1 - пропускаем Ollama фазу"
    return 1
  fi

  stop_local-27b
  docker rm -f vllm-bench 2>/dev/null || true
  log "Ollama on GPU1 (:11434)..."
  if systemctl is-active ollama >/dev/null 2>&1; then
    sudo systemctl stop ollama 2>/dev/null || systemctl stop ollama 2>/dev/null || true
    echo "systemd" >"$LOGDIR/ollama_stopped_via"
  fi
  pkill -x ollama 2>/dev/null || true
  sleep 2
  export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
  CUDA_VISIBLE_DEVICES=1 OLLAMA_MODELS="$OLLAMA_MODELS" \
    nohup /usr/local/bin/ollama serve >>"$LOGDIR/ollama_gpu1_fixed.log" 2>&1 &
  echo $! >"$LOGDIR/ollama_fixed.pid"

  # ИСПРАВЛЕНИЕ: увеличен timeout до 180 секунд (3 минуты)
  if ! wait_openai "http://127.0.0.1:11434/v1" "qwen3:8b" 180; then
    log "qwen3:8b not in catalog — pulling..."
    CUDA_VISIBLE_DEVICES=1 OLLAMA_MODELS="$OLLAMA_MODELS" ollama pull qwen3:8b \
      >>"$LOGDIR/ollama_gpu1_fixed.log" 2>&1 || true
    wait_openai "http://127.0.0.1:11434/v1" "qwen3:8b" 180
  fi
  log "Warming up qwen3:8b on GPU1..."
  CUDA_VISIBLE_DEVICES=1 OLLAMA_MODELS="$OLLAMA_MODELS" \
    ollama run qwen3:8b "2+2=4" >/dev/null 2>&1 || true
}

stop_local-27b() {
  # Останавливаем только если не local-27b ternary
  if [ -f "$LOGDIR/local-27b_server.pid" ]; then
    local pid=$(cat "$LOGDIR/local-27b_server.pid")
    if ! ps -p "$pid" -o cmd | grep -q "Ternary-local-27b-Q2_0.gguf"; then
      kill "$pid" 2>/dev/null || true
      rm -f "$LOGDIR/local-27b_server.pid"
    fi
  fi
  pkill -f 'llama-server.*local-27b' | grep -v "Ternary-local-27b-Q2_0.gguf" 2>/dev/null || true
  sleep 2
}

stop_ollama_bench() {
  if [ -f "$LOGDIR/ollama_fixed.pid" ]; then
    kill "$(cat "$LOGDIR/ollama_fixed.pid")" 2>/dev/null || true
    rm -f "$LOGDIR/ollama_fixed.pid"
  fi
  pkill -x ollama 2>/dev/null || true
  sleep 1
  if [ -f "$LOGDIR/ollama_stopped_via" ]; then
    sudo systemctl start ollama 2>/dev/null || systemctl start ollama 2>/dev/null || true
    rm -f "$LOGDIR/ollama_stopped_via"
  fi
}

run_eval() {
  local key="$1"
  local served="$2"
  local base_url="${3:-http://127.0.0.1:8077/v1}"
  local extra_flags="${4:-}"
  log "=== evaluate: $model_id @ $base_url (suite=SIMPLIFIED) ==="
  export HF_ALLOW_CODE_EVAL=1
  cd "$LLMBENCH"
  local thinking_flag="--no-thinking-body"
  if [ "$model_id" = "qwen2.5-vl-7b" ] || [ "$model_id" = "ollama-qwen3-8b" ] || [ "$model_id" = "glm-4-9b-chat" ]; then
    thinking_flag=""
  fi
  # shellcheck disable=SC2086
  .venv/bin/python scripts/evaluate_model.py \
    --model-key "$model_id" \
    --base-url "$base_url" \
    --served-model "$served" \
    --suite simplified \
    --context-scaling \
    --perf-n 12 \
    --perf-max-tokens 256 \
    $thinking_flag \
    --out-dir out \
    $extra_flags \
    2>&1 | tee -a "$LOGDIR/eval_${key}_fixed.log"
}

restore_services() {
  log "Restoring services..."
  stop_local-27b
  stop_ollama_bench
  docker rm -f vllm-bench 2>/dev/null || true
  (cd "$MLOPS" && docker compose --profile inference up -d triton) 2>&1 | tee -a "$MAIN_LOG" || true
  if ! pgrep -f 'ComfyUI.*main.py' >/dev/null 2>&1; then
    log "Restoring ComfyUI on GPU1..."
    cd /opt/llm-data/ComfyUI
    CUDA_VISIBLE_DEVICES=1 nohup ./venv/bin/python main.py \
      --listen 0.0.0.0 --port 8188 --lowvram --reserve-vram 2 \
      --use-pytorch-cross-attention \
      --output-directory /opt/llm-data/ComfyUI/output \
      --temp-directory /opt/llm-data/ComfyUI/temp \
      >>"$LOGDIR/comfyui_restart_fixed.log" 2>&1 &
  fi
}

trap 'log "INT/ERR — restoring services"; restore_services' EXIT

log "========== NIGHTLY COMPARE FIXED START (simplified suite, increased timeout) =========="
log "Installing lm-eval[math] if needed..."
cd "$LLMBENCH"
.venv/bin/pip install -q 'lm-eval[math]' huggingface_hub 2>&1 | tail -2 | tee -a "$MAIN_LOG" || true

log "GPU check:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a "$MAIN_LOG"

# Проверяем local-27b ternary
if pgrep -f "llama-server.*Ternary-local-27b-Q2_0.gguf" > /dev/null; then
  log "local-27b ternary активен на GPU1 - ПРОПУСКАЕМ LOCAL27B ФАЗУ"
  LOCAL27B_RUNNING=true
else
  log "local-27b ternary не найден - будем запускать позже"
  LOCAL27B_RUNNING=false
fi

stop_inference

# --- Phase 1: Qwen3-8B BF16 (HF cache), GPU0 ---
log "=== Phase 1: Qwen3-8B BF16 ==="
start_vllm "/root/.cache/huggingface/Qwen3-8B" "Qwen3-8B"
run_eval "qwen3-8b-bf16" "Qwen3-8B"
docker rm -f vllm-bench 2>/dev/null || true
sleep 8

# --- Phase 2: Qwen2.5-VL-7B (HF hub cache), GPU0 ---
log "=== Phase 2: Qwen2.5-VL-7B ==="
start_vllm "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct"
run_eval "qwen2.5-vl-7b" "Qwen2.5-VL-7B-Instruct"
docker rm -f vllm-bench 2>/dev/null || true
sleep 8

# --- Phase 2b: GLM-4-9b-chat (HF hub), GPU0 ---
log "=== Phase 2b: GLM-4-9b-chat ==="
start_vllm "THUDM/glm-4-9b-chat" "glm-4-9b-chat"
run_eval "glm-4-9b-chat" "glm-4-9b-chat"
docker rm -f vllm-bench 2>/dev/null || true
sleep 8

# --- Phase 3: Ollama qwen3:8b Q4, GPU1 (если local-27b ternary завершился) ---
log "=== Phase 3: Ollama qwen3:8b ==="
if start_ollama_gpu1; then
  run_eval "ollama-qwen3-8b" "qwen3:8b" "http://127.0.0.1:11434/v1"
  stop_ollama_bench
  sleep 3
else
  log "Ollama фаза пропущена - local-27b ternary еще работает или другие проблемы"
fi

# --- local-27b фаза ПРОПУЩЕНА (ternary работает, 1bit сломан) ---
log "=== local-27b phases SKIPPED (ternary running separately, 1bit broken) ==="

# --- Compare ---
log "=== compare_models ==="
cd "$LLMBENCH"
.venv/bin/python scripts/compare_models.py \
  --indir out \
  --baseline qwen3-8b-bf16 \
  --out-md out/compare.md \
  --out-json out/compare.json \
  2>&1 | tee -a "$MAIN_LOG"

trap - EXIT
restore_services
log "========== NIGHTLY COMPARE FIXED DONE =========="

# ИСПРАВЛЕНИЯ в config/models.yaml для simplified suite:
# Нужно добавить:
# simplified:
#   tasks: [gsm8k, minerva_math, mmlu_pro, humaneval]
#   limit: null
