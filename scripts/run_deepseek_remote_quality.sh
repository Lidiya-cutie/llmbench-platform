#!/bin/bash
# Remote DeepSeek-V4-Flash quality (lm-eval). Не трогает локальный GPU.
# Можно параллельно с GLM. Чекпоинт общий с after_glm_queue.
set -uo pipefail

LLMBENCH=/opt/llm-data/llmbench
LOGDIR="$LLMBENCH/logs"
OUT="$LLMBENCH/out"
CKPT="$LOGDIR/after_glm_queue.checkpoint"
MAIN="$LOGDIR/deepseek_remote_quality.log"
VENV_PY="$LLMBENCH/.venv/bin/python"
ENV_FILE="$LLMBENCH/config/deepseek_remote.env"
TASKS=(gsm8k minerva_math mmlu_pro humaneval)
MODEL_KEY=deepseek-v4-flash

mkdir -p "$LOGDIR" "$OUT" "$LLMBENCH/.cache"
# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY in $ENV_FILE}"
: "${DEEPSEEK_BASE_URL:=http://31.28.10.135:8000/v1}"
: "${DEEPSEEK_MODEL:=deepseek-v4-flash}"
export OPENAI_API_KEY HF_ALLOW_CODE_EVAL=1

log() { echo "[$(date -Is)] $*" | tee -a "$MAIN"; }
marked() { grep -qxF "$1" "$CKPT" 2>/dev/null; }
mark() {
  touch "$CKPT"
  grep -qxF "$1" "$CKPT" 2>/dev/null || echo "$1" >>"$CKPT"
}

task_done() {
  local root="$1" task="$2"
  local dir="$root/$task"
  [[ -d "$dir" ]] || return 1
  find "$dir" -name 'results*.json' 2>/dev/null | grep -q .
}

smoke() {
  log "smoke chat/completions"
  local code
  code=$(curl -sS -m 60 -o /tmp/ds_smoke.json -w '%{http_code}' \
    "${DEEPSEEK_BASE_URL%/}/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    -d "{\"model\":\"${DEEPSEEK_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8}")
  if [[ "$code" != "200" ]]; then
    log "ERROR: smoke http=$code body=$(head -c 200 /tmp/ds_smoke.json)"
    return 1
  fi
  log "smoke OK"
}

run_task() {
  local task="$1"
  local out_root="$OUT/lmeval_${MODEL_KEY}_queue"
  local tdir="$out_root/$task"
  mkdir -p "$tdir"
  log "lm-eval $MODEL_KEY task=$task @ $DEEPSEEK_BASE_URL"
  "$VENV_PY" -m lm_eval \
    --model local-chat-completions \
    --model_args "base_url=${DEEPSEEK_BASE_URL%/}/chat/completions,model=${DEEPSEEK_MODEL},num_concurrent=1,tokenized_requests=False" \
    --tasks "$task" \
    --apply_chat_template \
    --output_path "$tdir" \
    --log_samples \
    --use_cache "$LLMBENCH/.cache/lm_eval_${MODEL_KEY}_${task}" \
    --cache_requests true \
    --confirm_run_unsafe_code \
    >>"$MAIN" 2>&1
}

cd "$LLMBENCH"
log "=== DEEPSEEK REMOTE QUALITY START ==="

if marked "PHASE:${MODEL_KEY}" || marked "QUALITY:${MODEL_KEY}:ALL"; then
  log "SKIP $MODEL_KEY (checkpoint)"
  exit 0
fi

smoke || exit 1

out_root="$OUT/lmeval_${MODEL_KEY}_queue"
mkdir -p "$out_root"

for task in "${TASKS[@]}"; do
  tmark="QUALITY:${MODEL_KEY}:$task"
  if marked "$tmark" || task_done "$out_root" "$task"; then
    log "SKIP $MODEL_KEY/$task"
    mark "$tmark"
    continue
  fi
  if ! run_task "$task"; then
    log "ERROR: $MODEL_KEY/$task failed"
    exit 2
  fi
  if task_done "$out_root" "$task"; then
    mark "$tmark"
    log "DONE $MODEL_KEY/$task"
  else
    log "ERROR: no results $MODEL_KEY/$task"
    exit 3
  fi
done

"$VENV_PY" scripts/merge_lmeval_into_card.py \
  --card "$OUT/card_${MODEL_KEY}.json" \
  --lmeval-root "$out_root" \
  >>"$MAIN" 2>&1 || log "WARN: merge failed"

mark "QUALITY:${MODEL_KEY}:ALL"
mark "PHASE:${MODEL_KEY}"
log "=== DEEPSEEK REMOTE QUALITY COMPLETE ==="
exit 0
