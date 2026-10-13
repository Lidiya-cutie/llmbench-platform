#!/bin/bash
set -u
LLMBENCH=/opt/llm-data/llmbench
VENV=$LLMBENCH/.venv/bin/python
LOGDIR=$LLMBENCH/logs
OUT=$LLMBENCH/out/lmeval_glm-4-9b-chat_queue
MAIN=$LOGDIR/gpu1_glm_fill_gaps.log
BASE=http://127.0.0.1:8077/v1/chat/completions
MODEL=glm-4-9b-chat
export HF_ALLOW_CODE_EVAL=1

log(){ echo "[$(date -Is)] $*" | tee -a "$MAIN"; }
has(){ find "$1" -name 'results_*.json' 2>/dev/null | grep -q .; }
proxy_ok(){ curl -sf -m 5 http://127.0.0.1:8077/v1/models | grep -q glm; }
merge(){
  cd "$LLMBENCH"
  "$VENV" scripts/merge_lmeval_into_card.py --card out/card_glm-4-9b-chat.json --lmeval-root out/lmeval_glm-4-9b-chat_queue >>"$MAIN" 2>&1 || true
  "$VENV" - <<'PY' >>"$MAIN" 2>&1 || true
import json
from pathlib import Path
from datetime import datetime, timezone
card=json.load(open('out/card_glm-4-9b-chat.json'))
q=card.get('quality') or {}
bp=Path('out/dashboard_v2/bench_data.json')
b=json.load(open(bp))
for m in b['models']:
  if m['key']=='glm-4-9b-chat':
    for k in ['mmlu_pro','gpqa','gsm8k','minerva_math','humaneval','mbpp','ifeval']:
      m['scores'][k]=round(float(q.get(k) or 0),4)
b['generated_at']=datetime.now(timezone.utc).isoformat()
bp.write_text(json.dumps(b, ensure_ascii=False, indent=2))
print('glm bench refresh', {k:q.get(k) for k in ['humaneval','mbpp','ifeval','gpqa']})
PY
}

run_task(){
  local task="$1" out_sub="$2" conc="$3" max_gen="$4" extra_gen="${5:-}"
  local odir="$OUT/$out_sub"
  if has "$odir"; then log "SKIP $task"; return 0; fi
  proxy_ok || { log "GLM down before $task"; return 1; }
  log "START $task conc=$conc max_gen=$max_gen"
  mkdir -p "$odir"
  cd "$LLMBENCH"
  local gen_args=()
  if [[ -n "$extra_gen" ]]; then gen_args=(--gen_kwargs "$extra_gen"); fi
  "$VENV" -m lm_eval --model local-chat-completions \
    --model_args "model=${MODEL},base_url=${BASE},num_concurrent=${conc},max_length=4096,max_gen_toks=${max_gen},tokenized_requests=False" \
    --tasks "$task" --apply_chat_template --batch_size 1 \
    "${gen_args[@]}" \
    --confirm_run_unsafe_code \
    --output_path "$odir" --log_samples \
    >"$LOGDIR/glm_gpu1_${out_sub}.log" 2>&1
  local rc=$?
  log "$task exit=$rc has=$(has "$odir" && echo yes || echo no)"
  merge
  return $rc
}

log "=== GLM GPU1 GAP FILL ==="
# HumanEval instruct: need until=[] 
run_task humaneval_instruct humaneval_instruct 8 1024 'until=[]'
run_task mbpp_instruct mbpp_instruct 8 512 ''
# fallback mbpp if instruct missing results
if ! has "$OUT/mbpp_instruct"; then run_task mbpp mbpp 8 512 ''; fi
run_task ifeval ifeval 8 256 ''
run_task gpqa_main_generative_n_shot gpqa_main_generative_n_shot 8 256 ''
log "=== GLM GPU1 GAP FILL DONE ==="
