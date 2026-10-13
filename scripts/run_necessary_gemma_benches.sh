#!/bin/bash
set -u
LLMBENCH=/opt/llm-data/llmbench
VENV=$LLMBENCH/.venv/bin/python
LOGDIR=$LLMBENCH/logs
GOUT=$LLMBENCH/out/lmeval_gemma-3-27b-awq_queue
MAINLOG=$LOGDIR/necessary_benches_gemma.log
export HF_ALLOW_CODE_EVAL=1

log(){ echo "[$(date -Is)] $*" | tee -a "$MAINLOG"; }
has_results(){ find "$1" -name 'results_*.json' 2>/dev/null | grep -q .; }
proxy_ok(){ curl -sf -m 3 http://127.0.0.1:19892/v1/models | grep -q gemma; }
avail(){ awk '/MemAvailable:/{print int($2/1024)}' /proc/meminfo; }
wait_ram(){ while (( $(avail) < 6500 )); do log "RAM $(avail)"; sleep 30; done; }
merge(){ cd "$LLMBENCH" && "$VENV" scripts/merge_lmeval_into_card.py --card out/card_gemma-3-27b-awq.json --lmeval-root out/lmeval_gemma-3-27b-awq_queue >>"$MAINLOG" 2>&1 || true; }

log "=== NECESSARY GEMMA BENCHES START avail=$(avail)MB ==="
proxy_ok || { log "proxy down"; exit 1; }

MBPP_OUT=$GOUT/mbpp/gemma-3-27b-awq
if has_results "$MBPP_OUT"; then log "SKIP mbpp"; else
  wait_ram; proxy_ok || exit 1
  log "START mbpp conc=4"
  cd "$LLMBENCH"
  "$VENV" -m lm_eval --model local-chat-completions \
    --model_args "model=gemma-3-27b-awq,base_url=http://127.0.0.1:19892/v1/chat/completions,num_concurrent=4,max_length=3072,max_gen_toks=512,tokenized_requests=False" \
    --tasks mbpp --apply_chat_template --batch_size 1 --confirm_run_unsafe_code \
    --output_path "$MBPP_OUT" --log_samples \
    >"$LOGDIR/rerun_v4_mbpp_gemma_necessary.log" 2>&1
  log "mbpp exit=$? has_results=$(has_results "$MBPP_OUT" && echo yes || echo no)"
  merge
fi

MMLU_OUT=$GOUT/mmlu_pro/gemma-3-27b-awq
if has_results "$MMLU_OUT"; then log "SKIP mmlu_pro"; else
  wait_ram; proxy_ok || exit 1
  log "START mmlu_pro conc=4 (with cache)"
  mkdir -p "$MMLU_OUT"
  cd "$LLMBENCH"
  "$VENV" -m lm_eval --model local-chat-completions \
    --model_args "model=gemma-3-27b-awq,base_url=http://127.0.0.1:19892/v1/chat/completions,num_concurrent=4,max_length=3072,max_gen_toks=512,tokenized_requests=False" \
    --tasks mmlu_pro --apply_chat_template --batch_size 1 \
    --output_path "$MMLU_OUT" --log_samples \
    --use_cache "$LLMBENCH/.cache/lm_eval_gemma_mmlu_pro" \
    --cache_requests true \
    >"$LOGDIR/rerun_v4_mmlu_pro_gemma_necessary.log" 2>&1
  log "mmlu_pro exit=$? has_results=$(has_results "$MMLU_OUT" && echo yes || echo no)"
  merge
fi

GSM_OUT=$GOUT/gsm8k/gemma-3-27b-awq
if has_results "$GSM_OUT"; then log "SKIP gsm8k"; else
  wait_ram; proxy_ok || exit 1
  log "START gsm8k conc=4"
  mkdir -p "$GSM_OUT"
  cd "$LLMBENCH"
  "$VENV" -m lm_eval --model local-chat-completions \
    --model_args "model=gemma-3-27b-awq,base_url=http://127.0.0.1:19892/v1/chat/completions,num_concurrent=4,max_length=3072,max_gen_toks=512,tokenized_requests=False" \
    --tasks gsm8k --apply_chat_template --batch_size 1 \
    --output_path "$GSM_OUT" --log_samples \
    >"$LOGDIR/rerun_v4_gsm8k_gemma_necessary.log" 2>&1
  log "gsm8k exit=$? has_results=$(has_results "$GSM_OUT" && echo yes || echo no)"
  merge
fi

cd "$LLMBENCH" && "$VENV" scripts/generate_dashboard_v2.py >>"$MAINLOG" 2>&1 || true
log "=== NECESSARY GEMMA BENCHES COMPLETE avail=$(avail)MB ==="
