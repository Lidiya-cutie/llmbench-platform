#!/bin/bash
# Correct HumanEval for chat models: humaneval_instruct + no completion stops.
# Usage: run_humaneval_instruct_correct.sh <name> <model> <base_url> <out_dir> [concurrent]
set -u
NAME=${1:?name}
MODEL=${2:?model}
BASE_URL=${3:?base_url}
OUT=${4:?out_dir}
CONC=${5:-2}
LLMBENCH=/opt/llm-data/llmbench
VENV=$LLMBENCH/.venv/bin/python
LOG=$LLMBENCH/logs/humaneval_instruct_${NAME}.log
mkdir -p "$OUT" "$LLMBENCH/logs"
export HF_ALLOW_CODE_EVAL=1
export HF_ALLOW_CODE_EVAL=1

avail=$(awk '/MemAvailable:/{print int($2/1024)}' /proc/meminfo)
echo "[$(date -Is)] START humaneval_instruct name=$NAME model=$MODEL conc=$CONC avail=${avail}MB" | tee -a "$LOG"
if (( avail < 5500 )); then
  echo "[$(date -Is)] ABORT low RAM" | tee -a "$LOG"
  exit 1
fi

cd "$LLMBENCH"
"$VENV" -m lm_eval \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE_URL},num_concurrent=${CONC},max_length=4096,max_gen_toks=1024,tokenized_requests=False" \
  --tasks humaneval_instruct \
  --apply_chat_template \
  --batch_size 1 \
  --gen_kwargs 'until=[]' \
  --confirm_run_unsafe_code \
  --output_path "$OUT" \
  --log_samples \
  >>"$LOG" 2>&1
rc=$?
if find "$OUT" -name 'results_*.json' | grep -q .; then
  echo "[$(date -Is)] OK results present rc=$rc" | tee -a "$LOG"
  python3 - <<PY
import json,glob
files=sorted(glob.glob("$OUT/**/results_*.json", recursive=True))
d=json.load(open(files[-1]))
r=d["results"].get("humaneval_instruct") or d["results"].get("humaneval") or {}
print("[score]", {k:v for k,v in r.items() if "pass" in k and "stderr" not in k})
PY
else
  echo "[$(date -Is)] FAIL no results rc=$rc" | tee -a "$LOG"
fi
exit $rc
