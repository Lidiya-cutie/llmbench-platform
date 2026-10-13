#!/bin/bash
set -euo pipefail
LLMBENCH=/opt/llm-data/llmbench
VENV="$LLMBENCH/.venv/bin/python"
HF=/opt/llm-data/.cache_huggingface
OUT="$LLMBENCH/out/lmeval_qwen3-8b-bf16_queue"
LOG="$LLMBENCH/logs/qwen3_mmlu_pro_standalone.log"
mkdir -p "$OUT/mmlu_pro" "$LLMBENCH/.cache"
export HF_ALLOW_CODE_EVAL=1
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
cleanup() { log "cleanup: removing vllm-mmlu-pro container"; docker rm -f vllm-mmlu-pro 2>/dev/null || true; }
trap cleanup EXIT
log "Starting vLLM GPU1 :8078 — Qwen3-8B (BF16)"
docker rm -f vllm-mmlu-pro 2>/dev/null || true
docker run -d --name vllm-mmlu-pro --gpus '"device=1"' -v "$HF:/root/.cache/huggingface" -p 8078:8078 mlops-triton-vllm:24.12-qwen3 vllm serve "/root/.cache/huggingface/Qwen3-8B" --host 0.0.0.0 --port 8078 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 --trust-remote-code --served-model-name "Qwen3-8B"
log "Waiting for vLLM API on :8078..."
for i in $(seq 1 180); do
  if curl -sf -m 5 http://127.0.0.1:8078/v1/models 2>/dev/null | grep -q Qwen3; then log "API ready after ${i} checks"; break; fi
  if (( i % 12 == 0 )); then log "still waiting... ($i/180)"; fi
  sleep 5
done
if ! curl -sf -m 5 http://127.0.0.1:8078/v1/models 2>/dev/null | grep -q Qwen3; then log "ERROR: vLLM API not ready after 180 checks"; docker logs vllm-mmlu-pro 2>&1 | tail -30 >> "$LOG"; exit 1; fi
log "Starting lm_eval mmlu_pro — Qwen3-8B"
cd "$LLMBENCH"
"$VENV" -m lm_eval --model local-chat-completions --model_args "base_url=http://127.0.0.1:8078/v1/chat/completions,model=Qwen3-8B,num_concurrent=1,tokenized_requests=False" --tasks mmlu_pro --apply_chat_template --output_path "$OUT/mmlu_pro" --log_samples --use_cache "$LLMBENCH/.cache/lm_eval_qwen3-8b-bf16_mmlu_pro" --cache_requests true --confirm_run_unsafe_code 2>&1 | tee -a "$LOG"
if find "$OUT/mmlu_pro" -name 'results*.json' 2>/dev/null | grep -q .; then log "SUCCESS: results found"; else log "ERROR: no results in $OUT/mmlu_pro"; fi
log "=== DONE ==="
