#!/usr/bin/env bash
# Core-7 для test-моделей dashboard: internvl3-5-8b + janus-pro-7b.
# Не трогает index.html. Обновляет cards + out/dashboard_v2/test_models.json.
# GPU0: InternVL (после stop llmbench-vllm-gpu0 / GLM)
# GPU1: Janus text OpenAI-shim
set -uo pipefail

LLMBENCH=/opt/llm-data/llmbench
VENV=$LLMBENCH/.venv/bin/python
LOGDIR=$LLMBENCH/logs
MAIN=$LOGDIR/test_models_core7.log

INTERN_PORT=8071
INTERN_SERVED=InternVL3_5-8B-HF
INTERN_MODEL_PATH=/opt/llm-data/.cache_huggingface/hub/models--OpenGVLab--InternVL3_5-8B-HF/snapshots/741a7d03020411e666c6109218ab71e08151ef86
INTERN_OUT=$LLMBENCH/out/lmeval_internvl3-5-8b_queue
INTERN_CARD=$LLMBENCH/out/card_internvl3-5-8b.json
INTERN_SHIM=$LLMBENCH/scripts/internvl_openai_shim.py

JANUS_PORT=8072
JANUS_MODEL_PATH=/opt/llm-data/my_train_models/deepseek_and_colpali/Janus/model2
JANUS_SERVED=Janus-Pro-7B
JANUS_OUT=$LLMBENCH/out/lmeval_janus-pro-7b_queue
JANUS_CARD=$LLMBENCH/out/card_janus-pro-7b.json
JANUS_SHIM=$LLMBENCH/scripts/janus_openai_shim.py

CORE_TASKS=(gsm8k ifeval humaneval_instruct mbpp_instruct gpqa_main_generative_n_shot minerva_math mmlu_pro)

mkdir -p "$LOGDIR" "$INTERN_OUT" "$JANUS_OUT"
: >"$MAIN"
log(){ echo "[$(date -Is)] $*" | tee -a "$MAIN"; }

has_results(){ find "$1" -name 'results_20*.json' 2>/dev/null | grep -q .; }

wait_http(){
  local url="$1" expect="$2" tries="${3:-300}"
  local i
  for i in $(seq 1 "$tries"); do
    if curl -sf -m 5 "$url" 2>/dev/null | rg -q "$expect"; then
      log "ready $url ($i)"
      return 0
    fi
    if (( i % 20 == 0 )); then log "wait $url $i/$tries"; fi
    sleep 3
  done
  log "TIMEOUT $url"
  return 1
}

stop_glm_gpu0(){
  log "STOP llmbench-vllm-gpu0 (GLM already 7/7) to free GPU0"
  docker stop llmbench-vllm-gpu0 >>"$MAIN" 2>&1 || true
  sleep 4
}

start_internvl_shim(){
  docker rm -f vllm-internvl-bench >/dev/null 2>&1 || true
  if [[ -f "$LOGDIR/internvl_openai_shim.pid" ]]; then
    kill "$(cat "$LOGDIR/internvl_openai_shim.pid")" 2>/dev/null || true
  fi
  log "START InternVL transformers shim GPU0 :$INTERN_PORT"
  CUDA_VISIBLE_DEVICES=0 \
  INTERNVL_MODEL_PATH="$INTERN_MODEL_PATH" \
  INTERNVL_SERVED_NAME="$INTERN_SERVED" \
  INTERNVL_PORT="$INTERN_PORT" \
    nohup "$VENV" "$INTERN_SHIM" >"$LOGDIR/internvl_openai_shim.log" 2>&1 &
  echo $! >"$LOGDIR/internvl_openai_shim.pid"
  wait_http "http://127.0.0.1:${INTERN_PORT}/v1/models" "$INTERN_SERVED" 400
}

start_janus_shim(){
  "$VENV" -m pip install -q attrdict3 einops timm 2>>"$MAIN" || true
  if [[ -f "$LOGDIR/janus_openai_shim.pid" ]]; then
    kill "$(cat "$LOGDIR/janus_openai_shim.pid")" 2>/dev/null || true
  fi
  log "START Janus OpenAI shim GPU1 :$JANUS_PORT"
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=/opt/llm-data/my_train_models/deepseek_and_colpali/Janus \
  JANUS_MODEL_PATH="$JANUS_MODEL_PATH" \
  JANUS_SERVED_NAME="$JANUS_SERVED" \
  JANUS_PORT="$JANUS_PORT" \
    nohup "$VENV" "$JANUS_SHIM" >"$LOGDIR/janus_openai_shim.log" 2>&1 &
  echo $! >"$LOGDIR/janus_openai_shim.pid"
  wait_http "http://127.0.0.1:${JANUS_PORT}/v1/models" "$JANUS_SERVED" 400
}

run_task(){
  local tag="$1" task="$2" out="$3" base="$4" model="$5" max_gen="$6" conc="$7"
  local tdir="$out/$task"
  mkdir -p "$tdir"
  if has_results "$tdir"; then log "SKIP $tag/$task"; return 0; fi
  log "START $tag/$task"
  cd "$LLMBENCH"
  export HF_ALLOW_CODE_EVAL=1
  local args=(
    -m lm_eval --model local-chat-completions
    --model_args "model=${model},base_url=${base}/v1/chat/completions,num_concurrent=${conc},max_length=8192,max_gen_toks=${max_gen},tokenized_requests=False"
    --tasks "$task" --apply_chat_template --batch_size 1
    --confirm_run_unsafe_code
    --output_path "$tdir" --log_samples
  )
  if [[ "$task" == humaneval_instruct ]]; then
    args+=(--gen_kwargs 'until=[]')
  fi
  "$VENV" "${args[@]}" >"$LOGDIR/${tag}_${task}.log" 2>&1
  local rc=$?
  if has_results "$tdir"; then log "OK $tag/$task rc=$rc"; else
    log "FAIL $tag/$task rc=$rc"; tail -25 "$LOGDIR/${tag}_${task}.log" | tee -a "$MAIN"; fi
  return $rc
}

run_model_queue(){
  local tag="$1" base="$2" model="$3" out="$4"
  # max_gen / conc tuned per task
  # conc=1: transformers shims serialize generate under a lock
  run_task "$tag" gsm8k "$out" "$base" "$model" 256 1 || true
  run_task "$tag" ifeval "$out" "$base" "$model" 1280 1 || true
  run_task "$tag" humaneval_instruct "$out" "$base" "$model" 1024 1 || true
  run_task "$tag" mbpp_instruct "$out" "$base" "$model" 512 1 || true
  run_task "$tag" gpqa_main_generative_n_shot "$out" "$base" "$model" 256 1 || true
  run_task "$tag" minerva_math "$out" "$base" "$model" 512 1 || true
  run_task "$tag" mmlu_pro "$out" "$base" "$model" 256 1 || true
}

offline_fix_coding(){
  local tag="$1" out="$2"
  local he_s mb_s
  he_s=$(find "$out/humaneval_instruct" -name 'samples_*.jsonl' 2>/dev/null | sort | tail -1 || true)
  mb_s=$(find "$out/mbpp_instruct" -name 'samples_*.jsonl' 2>/dev/null | sort | tail -1 || true)
  if [[ -n "$he_s" ]]; then
    log "offline rescore HE $tag"
    "$VENV" "$LLMBENCH/scripts/offline_rescore_humaneval.py" --samples "$he_s" --out-dir "$(dirname "$he_s")" >>"$MAIN" 2>&1 || true
  fi
  if [[ -n "$mb_s" ]]; then
    log "offline rescore MBPP $tag"
    "$VENV" "$LLMBENCH/scripts/offline_rescore_mbpp.py" --samples "$mb_s" --task-key mbpp_instruct --out-dir "$(dirname "$mb_s")" >>"$MAIN" 2>&1 || true
  fi
}

merge_and_dashboard(){
  log "MERGE cards + test_models.json"
  "$VENV" "$LLMBENCH/scripts/merge_lmeval_into_card.py" --card "$INTERN_CARD" --lmeval-root "$INTERN_OUT" >>"$MAIN" 2>&1 || true
  "$VENV" "$LLMBENCH/scripts/merge_lmeval_into_card.py" --card "$JANUS_CARD" --lmeval-root "$JANUS_OUT" >>"$MAIN" 2>&1 || true
  "$VENV" "$LLMBENCH/scripts/sync_test_models_from_cards.py" >>"$MAIN" 2>&1 || true
}

cleanup(){
  log "cleanup servers (keep results)"
  docker rm -f vllm-internvl-bench >/dev/null 2>&1 || true
  if [[ -f "$LOGDIR/internvl_openai_shim.pid" ]]; then
    kill "$(cat "$LOGDIR/internvl_openai_shim.pid")" 2>/dev/null || true
  fi
  if [[ -f "$LOGDIR/janus_openai_shim.pid" ]]; then
    kill "$(cat "$LOGDIR/janus_openai_shim.pid")" 2>/dev/null || true
  fi
}
trap cleanup EXIT

main(){
  log "=== test models core-7: internvl + janus (transformers shims) ==="
  stop_glm_gpu0
  start_internvl_shim || { log "InternVL shim failed"; exit 1; }
  start_janus_shim || { log "Janus shim failed"; exit 1; }

  (
    run_model_queue internvl "http://127.0.0.1:${INTERN_PORT}" "$INTERN_SERVED" "$INTERN_OUT"
    offline_fix_coding internvl "$INTERN_OUT"
    log "internvl queue done"
  ) &
  P1=$!

  (
    run_model_queue janus "http://127.0.0.1:${JANUS_PORT}" "$JANUS_SERVED" "$JANUS_OUT"
    offline_fix_coding janus "$JANUS_OUT"
    log "janus queue done"
  ) &
  P2=$!

  wait "$P1" || true
  wait "$P2" || true
  merge_and_dashboard
  log "DONE"
  "$VENV" - <<'PY' | tee -a "$MAIN"
import json
from pathlib import Path
CORE=["mmlu_pro","gsm8k","minerva_math","humaneval","mbpp","ifeval","gpqa"]
for name in ["internvl3-5-8b","janus-pro-7b"]:
  c=json.load(open(f"/opt/llm-data/llmbench/out/card_{name}.json"))
  q=c.get("quality") or {}
  gaps=[k for k in CORE if not (isinstance(q.get(k),(int,float)) and float(q[k])>0)]
  print(name, f"{7-len(gaps)}/7", {k:q.get(k) for k in CORE}, "gaps", gaps)
PY
}

main "$@"
