#!/bin/bash
# Resumable GLM simplified bench: one lm-eval task group at a time + checkpoint.
# Survives host reboot if systemd restarts this unit (completed tasks are skipped).
set -uo pipefail

LLMBENCH=/opt/llm-data/llmbench
LOGDIR="$LLMBENCH/logs"
OUT="$LLMBENCH/out"
CKPT="$LOGDIR/glm_bench.checkpoint"
MAIN="$LOGDIR/resume_glm_gpu1.log"
VENV_PY="$LLMBENCH/.venv/bin/python"
BASE_URL="${BASE_URL:-http://127.0.0.1:8077/v1}"
SERVED="${SERVED_MODEL:-zai-org/glm-4-9b-chat}"
LMEVAL_ROOT="$OUT/lmeval_glm_resumable"
TASKS=(gsm8k minerva_math mmlu_pro humaneval)

mkdir -p "$LOGDIR" "$LMEVAL_ROOT"
export HF_ALLOW_CODE_EVAL=1

log() { echo "[$(date -Is)] $*" | tee -a "$MAIN"; }

wait_tgi() {
  local i
  for i in $(seq 1 120); do
    if curl -sf -m 3 "$BASE_URL/models" >/dev/null 2>&1 || curl -sf -m 3 http://127.0.0.1:8077/health >/dev/null 2>&1; then
      log "TGI ready (${i})"
      return 0
    fi
    sleep 5
  done
  log "ERROR: TGI not ready"
  return 1
}

task_done() {
  local task="$1"
  local dir="$LMEVAL_ROOT/$task"
  [[ -d "$dir" ]] || return 1
  find "$dir" -name 'results*.json' 2>/dev/null | grep -q .
}

mark_done() {
  local task="$1"
  mkdir -p "$(dirname "$CKPT")"
  touch "$CKPT"
  grep -qxF "$task" "$CKPT" 2>/dev/null || echo "$task" >> "$CKPT"
}

run_one_task() {
  local task="$1"
  local tdir="$LMEVAL_ROOT/$task"
  mkdir -p "$tdir"
  log "lm-eval task=$task -> $tdir"
  # shellcheck disable=SC2086
  mkdir -p "$LLMBENCH/.cache"
  "$VENV_PY" -m lm_eval \
    --model local-chat-completions \
    --model_args "base_url=${BASE_URL%/}/chat/completions,model=${SERVED},num_concurrent=1,tokenized_requests=False" \
    --tasks "$task" \
    --apply_chat_template \
    --output_path "$tdir" \
    --log_samples \
    --use_cache "$LLMBENCH/.cache/lm_eval_${task}" \
    --cache_requests true \
    --confirm_run_unsafe_code \
    >>"$MAIN" 2>&1
}

cd "$LLMBENCH"
log "=== RESUMABLE glm simplified START ==="
wait_tgi || exit 1

# micro+perf once if not marked
if ! grep -qxF "micro_perf" "$CKPT" 2>/dev/null; then
  log "micro+perf+context-scaling"
  "$VENV_PY" scripts/evaluate_model.py \
    --model-key glm-4-9b-chat \
    --base-url "$BASE_URL" \
    --served-model "$SERVED" \
    --perf-n 12 \
    --perf-max-tokens 256 \
    --context-scaling \
    --out-dir "$OUT" \
    >>"$MAIN" 2>&1 || log "WARN: micro/perf failed (continuing to lm-eval)"
  mark_done "micro_perf"
else
  log "SKIP micro_perf (checkpoint)"
fi

for task in "${TASKS[@]}"; do
  if task_done "$task" || grep -qxF "$task" "$CKPT" 2>/dev/null; then
    log "SKIP $task (already done)"
    mark_done "$task"
    continue
  fi
  if ! run_one_task "$task"; then
    log "ERROR: task $task failed (will retry on next start)"
    exit 2
  fi
  if task_done "$task"; then
    mark_done "$task"
    log "DONE $task"
  else
    log "ERROR: no results for $task"
    exit 3
  fi
done

log "merge quality into card via evaluate_model quality-only path / compare"
# Rebuild card from latest partial results by re-running finalize helper if present;
# fallback: compare_models on existing cards after writing a stub quality json.
"$VENV_PY" - <<'PY' >>"$MAIN" 2>&1 || true
import glob, json, os
from pathlib import Path
from llmbench.lmeval import parse_results_file, _pick_primary

root = Path("/opt/llm-data/llmbench/out/lmeval_glm_resumable")
primary = {}
tasks = {}
for task_dir in sorted(root.iterdir() if root.exists() else []):
    if not task_dir.is_dir():
        continue
    files = sorted(task_dir.rglob("results*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        continue
    le = parse_results_file(str(files[-1]))
    if not le.ok:
        continue
    primary.update(le.primary)
    tasks.update(le.tasks)

card_path = Path("/opt/llm-data/llmbench/out/card_glm-4-9b-chat.json")
if card_path.exists():
    card = json.loads(card_path.read_text())
else:
    card = {"model_key": "glm-4-9b-chat", "quality": {}}
card.setdefault("quality", {})
card["quality"].update(primary)
card["quality_detail"] = tasks
card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2))
print("updated card quality keys:", sorted(primary))
PY

"$VENV_PY" scripts/compare_models.py --indir "$OUT" --baseline qwen3-8b-bf16 \
  --out-md "$OUT/compare.md" --out-json "$OUT/compare.json" >>"$MAIN" 2>&1 || true

mark_done "ALL"
log "=== RESUMABLE glm simplified COMPLETE ==="
