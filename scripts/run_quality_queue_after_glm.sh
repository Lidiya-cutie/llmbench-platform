#!/bin/bash
# Очередь quality (lm-eval) ПОСЛЕ успешного завершения llmbench-glm-resumable.
# Не конкурирует с текущим GLM: ждёт inactive; GPU1 освобождает только после GLM.
# Resumable: чекпоинт + systemd Restart=on-failure.
set -uo pipefail

LLMBENCH=/opt/llm-data/llmbench
LOGDIR="$LLMBENCH/logs"
OUT="$LLMBENCH/out"
CKPT="$LOGDIR/after_glm_queue.checkpoint"
MAIN="$LOGDIR/after_glm_queue.log"
VENV_PY="$LLMBENCH/.venv/bin/python"
HF_CACHE=/opt/llm-data/.cache_huggingface
VLLM_IMAGE=mlops-triton-vllm:24.12-qwen3
TASKS=(gsm8k minerva_math mmlu_pro humaneval)
GLM_UNIT=llmbench-glm-resumable.service

mkdir -p "$LOGDIR" "$OUT"
export HF_ALLOW_CODE_EVAL=1

log() { echo "[$(date -Is)] $*" | tee -a "$MAIN"; }

marked() { grep -qxF "$1" "$CKPT" 2>/dev/null; }
mark() {
  touch "$CKPT"
  grep -qxF "$1" "$CKPT" 2>/dev/null || echo "$1" >>"$CKPT"
}

wait_glm_done() {
  log "WAIT: $GLM_UNIT ..."
  local n=0
  while true; do
    n=$((n + 1))
    if systemctl is-active --quiet "$GLM_UNIT"; then
      if (( n % 12 == 1 )); then
        log "GLM still active (check #$n) — sleep 300s"
      fi
      sleep 300
      continue
    fi
    if systemctl is-failed --quiet "$GLM_UNIT"; then
      log "ERROR: $GLM_UNIT failed — очередь НЕ запускается"
      return 1
    fi
    if marked "ALL" && grep -qxF "ALL" "$LOGDIR/glm_bench.checkpoint" 2>/dev/null; then
      :
    fi
    # inactive + not-failed
    if grep -qxF "ALL" "$LOGDIR/glm_bench.checkpoint" 2>/dev/null; then
      log "GLM checkpoint ALL present"
    else
      log "WARN: glm checkpoint without ALL — сервис inactive; продолжаем осторожно"
    fi
    log "GLM finished (inactive, not-failed)"
    return 0
  done
}

wait_http() {
  local url="$1" expect="${2:-}" tries="${3:-120}"
  local i resp
  for i in $(seq 1 "$tries"); do
    if resp=$(curl -sf -m 5 "$url" 2>/dev/null); then
      if [[ -z "$expect" ]] || echo "$resp" | grep -q "$expect"; then
        log "API ready: $url ($i)"
        return 0
      fi
    fi
    sleep 5
  done
  log "ERROR: API not ready $url expect=$expect"
  return 1
}

stop_tgi_glm() {
  if marked "STOP_TGI"; then
    log "SKIP STOP_TGI"
    return 0
  fi
  log "Stopping tgi-glm to free GPU1 (triton on GPU0 не трогаем)"
  docker stop tgi-glm 2>/dev/null || true
  sleep 5
  mark "STOP_TGI"
}

stop_vllm() {
  docker rm -f vllm-bench-queue 2>/dev/null || true
  sleep 3
}

start_vllm_gpu1() {
  local model_path="$1" served="$2"
  stop_vllm
  log "vLLM GPU1 :8077 — $served ($model_path)"
  docker run -d --name vllm-bench-queue \
    --gpus '"device=1"' \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -p 8077:8077 \
    "$VLLM_IMAGE" \
    vllm serve "$model_path" \
    --host 0.0.0.0 --port 8077 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --served-model-name "$served"
  wait_http "http://127.0.0.1:8077/v1/models" "$served" 180
}

stop_local-27b() {
  pkill -f 'llama-server.*Ternary-local-27b|llama-server.*local-27b' 2>/dev/null || true
  sleep 2
}

start_local-27b_ternary_gpu1() {
  stop_local-27b
  stop_vllm
  local MODEL_PATH="/usr/share/ollama/.ollama/models/Ternary-local-27b-gguf/Q2_0/Ternary-local-27b-Q2_0-6.66.gguf"
  if [[ ! -f "$MODEL_PATH" ]]; then
    log "ERROR: local-27b model missing: $MODEL_PATH"
    return 1
  fi
  log "local-27b ternary GPU1 :8080"
  (
    cd /opt/llm-data/local-27b-demo
    export CUDA_VISIBLE_DEVICES=1 LOCAL27B_THINKING=0
    nohup ./bin/cuda/llama-server \
      --model "$MODEL_PATH" \
      --host 127.0.0.1 --port 8080 \
      --ctx-size 8192 --n-gpu-layers 999 --parallel 1 \
      >>"$LOGDIR/local-27b_ternary_queue.log" 2>&1 &
    echo $! >"$LOGDIR/local-27b_ternary_queue.pid"
  )
  wait_http "http://127.0.0.1:8080/v1/models" "" 90 || wait_http "http://127.0.0.1:8080/health" "" 30
}

task_done() {
  local root="$1" task="$2"
  local dir="$root/$task"
  [[ -d "$dir" ]] || return 1
  find "$dir" -name 'results*.json' 2>/dev/null | grep -q .
}

run_lmeval_task() {
  local base_url="$1" served="$2" task="$3" out_root="$4" cache_tag="$5"
  local tdir="$out_root/$task"
  mkdir -p "$tdir" "$LLMBENCH/.cache"
  log "lm-eval $cache_tag task=$task"
  "$VENV_PY" -m lm_eval \
    --model local-chat-completions \
    --model_args "base_url=${base_url%/}/chat/completions,model=${served},num_concurrent=1,tokenized_requests=False" \
    --tasks "$task" \
    --apply_chat_template \
    --output_path "$tdir" \
    --log_samples \
    --use_cache "$LLMBENCH/.cache/lm_eval_${cache_tag}_${task}" \
    --cache_requests true \
    --confirm_run_unsafe_code \
    >>"$MAIN" 2>&1
}

run_quality_suite() {
  local model_key="$1" base_url="$2" served="$3"
  local out_root="$OUT/lmeval_${model_key}_queue"
  local mark_prefix="QUALITY:$model_key"
  mkdir -p "$out_root"

  if marked "${mark_prefix}:ALL"; then
    log "SKIP $model_key quality (checkpoint)"
    return 0
  fi

  for task in "${TASKS[@]}"; do
    local tmark="${mark_prefix}:$task"
    if marked "$tmark" || task_done "$out_root" "$task"; then
      log "SKIP $model_key/$task"
      mark "$tmark"
      continue
    fi
    if ! run_lmeval_task "$base_url" "$served" "$task" "$out_root" "$model_key"; then
      log "ERROR: $model_key/$task failed"
      return 2
    fi
    if task_done "$out_root" "$task"; then
      mark "$tmark"
      log "DONE $model_key/$task"
    else
      log "ERROR: no results $model_key/$task"
      return 3
    fi
  done

  log "merge quality -> card_$model_key"
  "$VENV_PY" scripts/merge_lmeval_into_card.py \
    --card "$OUT/card_${model_key}.json" \
    --lmeval-root "$out_root" \
    >>"$MAIN" 2>&1 || log "WARN: merge failed for $model_key"

  mark "${mark_prefix}:ALL"
  return 0
}

finalize_glm() {
  if marked "FINALIZE_GLM"; then
    log "SKIP FINALIZE_GLM"
    return 0
  fi
  log "Finalize GLM card + compare"
  "$VENV_PY" scripts/merge_lmeval_into_card.py \
    --card "$OUT/card_glm-4-9b-chat.json" \
    --lmeval-root "$OUT/lmeval_glm_resumable" \
    --rebuild-compare-preview \
    >>"$MAIN" 2>&1 || true
  "$VENV_PY" scripts/compare_models.py \
    --indir "$OUT" --baseline qwen3-8b-bf16 \
    --out-md "$OUT/compare.md" --out-json "$OUT/compare.json" \
    >>"$MAIN" 2>&1 || true
  mark "FINALIZE_GLM"
}

phase_deepseek_remote() {
  # remote API — без локального GPU; 1bit/InternVL сюда не добавлять
  if [[ "${SKIP_DEEPSEEK_REMOTE:-0}" == "1" ]]; then
    log "SKIP phase deepseek remote (SKIP_DEEPSEEK_REMOTE=1)"
    mark "PHASE:deepseek-v4-flash"
    return 0
  fi
  marked "PHASE:deepseek-v4-flash" && { log "SKIP phase deepseek remote"; return 0; }
  if [[ -x "$LLMBENCH/scripts/run_deepseek_remote_quality.sh" ]]; then
    log "Delegating DeepSeek remote quality script"
    bash "$LLMBENCH/scripts/run_deepseek_remote_quality.sh" || return $?
  else
    log "ERROR: missing run_deepseek_remote_quality.sh"
    return 1
  fi
}

phase_qwen3() {
  marked "PHASE:qwen3-8b-bf16" && { log "SKIP phase qwen3"; return 0; }
  start_vllm_gpu1 "/root/.cache/huggingface/Qwen3-8B" "Qwen3-8B" || return 1
  run_quality_suite "qwen3-8b-bf16" "http://127.0.0.1:8077/v1" "Qwen3-8B" || return $?
  stop_vllm
  mark "PHASE:qwen3-8b-bf16"
}

phase_qwen25vl() {
  marked "PHASE:qwen2.5-vl-7b" && { log "SKIP phase qwen2.5-vl"; return 0; }
  start_vllm_gpu1 "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct" || return 1
  run_quality_suite "qwen2.5-vl-7b" "http://127.0.0.1:8077/v1" "Qwen2.5-VL-7B-Instruct" || return $?
  stop_vllm
  mark "PHASE:qwen2.5-vl-7b"
}

phase_ollama() {
  marked "PHASE:ollama-qwen3-8b" && { log "SKIP phase ollama"; return 0; }
  stop_vllm
  stop_local-27b
  # локальный ollama уже слушает :11434 с qwen3:8b — не перезапускаем
  if ! curl -sf -m 5 http://127.0.0.1:11434/api/tags | grep -q 'qwen3:8b'; then
    log "ERROR: local ollama without qwen3:8b"
    return 1
  fi
  wait_http "http://127.0.0.1:11434/v1/models" "qwen3" 60 || return 1
  run_quality_suite "ollama-qwen3-8b" "http://127.0.0.1:11434/v1" "qwen3:8b" || return $?
  mark "PHASE:ollama-qwen3-8b"
}

phase_local-27b() {
  marked "PHASE:local-27b-27b-ternary" && { log "SKIP phase local-27b"; return 0; }
  start_local-27b_ternary_gpu1 || return 1
  # served name в llama-server часто произвольный — пробуем local-27b / первый model id
  local served="local-27b"
  if resp=$(curl -sf http://127.0.0.1:8080/v1/models 2>/dev/null); then
    served=$(echo "$resp" | "$VENV_PY" -c 'import sys,json; d=json.load(sys.stdin); print((d.get("data") or [{"id":"local-27b"}])[0].get("id","local-27b"))' 2>/dev/null || echo local-27b)
  fi
  log "local-27b served_model=$served"
  run_quality_suite "local-27b-27b-ternary" "http://127.0.0.1:8080/v1" "$served" || return $?
  stop_local-27b
  mark "PHASE:local-27b-27b-ternary"
}

final_compare() {
  if marked "COMPARE_FINAL"; then
    log "SKIP COMPARE_FINAL"
    return 0
  fi
  log "Final compare_models"
  "$VENV_PY" scripts/compare_models.py \
    --indir "$OUT" --baseline qwen3-8b-bf16 --ranking \
    --out-md "$OUT/compare.md" --out-json "$OUT/compare.json" \
    >>"$MAIN" 2>&1 || true
  mark "COMPARE_FINAL"
}

cleanup_soft() {
  log "soft cleanup: stop queue servers (tgi-glm already stopped if phase ran)"
  stop_vllm
  stop_local-27b
}

# --- main ---
cd "$LLMBENCH"
log "=== AFTER-GLM QUALITY QUEUE START ==="

if marked "ALL"; then
  log "Queue already ALL — nothing to do"
  exit 0
fi

wait_glm_done || exit 1
finalize_glm

# remote baseline можно до/вместо ожидания GPU — здесь после GLM finalize
phase_deepseek_remote || log "WARN: deepseek remote phase failed (GPU phases continue)"

stop_tgi_glm

phase_qwen3 || { cleanup_soft; exit 2; }
phase_qwen25vl || { cleanup_soft; exit 2; }
phase_ollama || { cleanup_soft; exit 2; }
phase_local-27b || { cleanup_soft; exit 2; }
# local-27b-1bit и internvl3_5 — НЕ в очереди (1bit проблемный; InternVL не готов)

final_compare
cleanup_soft
mark "ALL"
log "=== AFTER-GLM QUALITY QUEUE COMPLETE ==="
log "NOTE: tgi-glm остановлен; при необходимости: docker start tgi-glm"
exit 0
