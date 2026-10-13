#!/bin/bash
# Qwen3 MBPP без thinking — иначе max_tokens съедается открытым <think> → pass@1=0.
set -u
LLMBENCH=/opt/llm-data/llmbench
VENV=$LLMBENCH/.venv/bin/python
OUT=$LLMBENCH/out/lmeval_qwen3-8b-bf16_queue/mbpp_instruct/Qwen3-8B
LOG=$LLMBENCH/logs/rerun_v4_mbpp_instruct_qwen3_nothink.log
BASE_URL=${1:-http://127.0.0.1:8078/v1/chat/completions}
CONC=${2:-4}

mkdir -p "$OUT" "$LLMBENCH/logs"
export HF_ALLOW_CODE_EVAL=1

if ! curl -sf -m 5 http://127.0.0.1:8078/v1/models | grep -q Qwen3; then
  echo "[$(date -Is)] ABORT Qwen endpoint down" | tee -a "$LOG"
  exit 1
fi

avail=$(awk '/MemAvailable:/{print int($2/1024)}' /proc/meminfo)
echo "[$(date -Is)] START mbpp_instruct nothink conc=$CONC avail=${avail}MB" | tee -a "$LOG"
cd "$LLMBENCH"
"$VENV" -m lm_eval \
  --model local-chat-completions \
  --model_args "model=Qwen3-8B,base_url=${BASE_URL},num_concurrent=${CONC},max_length=4096,max_gen_toks=1024,tokenized_requests=False" \
  --tasks mbpp_instruct \
  --apply_chat_template \
  --batch_size 1 \
  --gen_kwargs '{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  --confirm_run_unsafe_code \
  --output_path "$OUT" \
  --log_samples \
  >>"$LOG" 2>&1
rc=$?
echo "[$(date -Is)] EXIT rc=$rc" | tee -a "$LOG"
if find "$OUT" -name 'results_*.json' | grep -q .; then
  "$VENV" - <<'PY'
import json,glob
files=sorted(glob.glob("/opt/llm-data/llmbench/out/lmeval_qwen3-8b-bf16_queue/mbpp_instruct/**/results_*.json", recursive=True))
d=json.load(open(files[-1]))
r=d["results"].get("mbpp_instruct") or d["results"].get("mbpp") or {}
print("[score]", {k:v for k,v in r.items() if isinstance(v,(int,float)) and "stderr" not in k})
PY
fi
exit $rc
