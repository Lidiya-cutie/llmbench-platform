#!/bin/bash
set -euo pipefail
LLMBENCH=/opt/llmbench
MLOPS=/opt/mlops-stack
LOGDIR="$LLMBENCH/logs"
MAIN="$LOGDIR/resume_glm_simplified.log"
mkdir -p "$LOGDIR"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$MAIN"; }

cleanup() {
  log "cleanup: remove vllm-bench, restore triton"
  docker rm -f vllm-bench 2>/dev/null || true
  (cd "$MLOPS" && docker compose --profile inference up -d triton) 2>/dev/null || true
}
trap cleanup EXIT

log "=== RESUME glm-4-9b-chat simplified ==="
log "Stop Triton, free GPU0"
(cd "$MLOPS" && docker compose --profile inference stop triton) 2>&1 | tee -a "$MAIN" || true
docker rm -f vllm-bench 2>/dev/null || true
fuser -k 8077/tcp 2>/dev/null || true
sleep 3

log "Start vLLM GLM on GPU0 :8077"
docker run -d --name vllm-bench \
  --gpus device=0 \
  -v /opt/hf-cache:/root/.cache/huggingface \
  -p 8077:8077 \
  mlops-triton-vllm:24.12-qwen3 \
  vllm serve THUDM/glm-4-9b-chat \
  --host 0.0.0.0 --port 8077 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --served-model-name glm-4-9b-chat

for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8077/v1/models 2>/dev/null | grep -q glm-4-9b-chat; then
    log "API ready (${i}*5s)"
    break
  fi
  sleep 5
  if [ "$i" -eq 120 ]; then log "API NOT READY"; exit 1; fi
  [ $((i % 12)) -eq 0 ] && log "waiting API $i/120..."
done

export HF_ALLOW_CODE_EVAL=1
cd "$LLMBENCH"
log "evaluate_model simplified..."
.venv/bin/python scripts/evaluate_model.py \
  --model-key glm-4-9b-chat \
  --base-url http://127.0.0.1:8077/v1 \
  --served-model glm-4-9b-chat \
  --suite simplified \
  --context-scaling \
  --perf-n 12 \
  --perf-max-tokens 256 \
  --out-dir out \
  2>&1 | tee -a "$LOGDIR/eval_glm-4-9b-chat.log"

log "rebuild compare + decision cards"
.venv/bin/python scripts/compare_models.py --indir out --baseline qwen3-8b-bf16 \
  --out-md out/compare.md --out-json out/compare.json 2>&1 | tee -a "$MAIN" || true
log "=== RESUME DONE ==="
