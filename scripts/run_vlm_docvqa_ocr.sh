#!/usr/bin/env bash
# Масштабируемый Vision/OCR прогон на свободном GPU (по умолчанию GPU1).
# Не трогает Triton/GPU0. Полный DocVQA/OCRBench только при LLMBENCH_VL_EXECUTE=1.
#
# Пример (только adapter smoke, без датасетов):
#   MODEL_CARD=qwen2.5-vl-7b BASE_URL=http://127.0.0.1:8077 \
#   SERVED_MODEL=Qwen2.5-VL-7B-Instruct bash scripts/run_vlm_docvqa_ocr.sh
#
# Полный прогон (после setup_vlmeval_venv.sh и поднятого VLM на GPU1):
#   LLMBENCH_VL_EXECUTE=1 TASKS=docvqa,ocrbench GPU_ID=1 \
#   MODEL_CARD=qwen2.5-vl-7b BASE_URL=http://127.0.0.1:8077 \
#   SERVED_MODEL=Qwen2.5-VL-7B-Instruct bash scripts/run_vlm_docvqa_ocr.sh
set -euo pipefail

LLMBENCH="${LLMBENCH:-/opt/llm-data/llmbench}"
VENV_PY="$LLMBENCH/.venv/bin/python"
GPU_ID="${GPU_ID:-1}"
MODEL_CARD="${MODEL_CARD:-qwen2.5-vl-7b}"
SERVED_MODEL="${SERVED_MODEL:-Qwen2.5-VL-7B-Instruct}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8077}"
TASKS="${TASKS:-docvqa,ocrbench}"
OUT="${OUT:-$LLMBENCH/out/vlmeval_${MODEL_CARD}_queue}"
LOGDIR="$LLMBENCH/logs"
MAIN="$LOGDIR/vlm_${MODEL_CARD}_docvqa_ocr.log"
CARD="$LLMBENCH/out/card_${MODEL_CARD}.json"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
EXECUTE="${LLMBENCH_VL_EXECUTE:-0}"

mkdir -p "$LOGDIR" "$OUT"
log(){ echo "[$(date -Is)] $*" | tee -a "$MAIN"; }

gpu_free(){
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' '
}

assert_safe(){
  local free
  free=$(gpu_free)
  log "GPU${GPU_ID} free=${free} MiB execute=${EXECUTE}"
  if [[ "$EXECUTE" == "1" || "$EXECUTE" == "true" ]]; then
    if (( free < MIN_FREE_MIB )); then
      log "ABORT: GPU${GPU_ID} not free enough ($free < $MIN_FREE_MIB)"
      exit 1
    fi
  fi
  if [[ -f "$CARD" ]]; then
    log "card=$CARD"
  else
    log "ABORT: missing $CARD"
    exit 1
  fi
}

run_eval(){
  cd "$LLMBENCH"
  IFS=',' read -r -a task_arr <<< "$TASKS"
  local py_tasks
  py_tasks=$(printf "'%s'," "${task_arr[@]}")
  py_tasks="[${py_tasks%,}]"
  log "START vision tasks=$TASKS out=$OUT base=$BASE_URL model=$SERVED_MODEL"
  LLMBENCH_VL_EXECUTE="$EXECUTE" "$VENV_PY" - <<PY | tee -a "$MAIN"
import importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, "$LLMBENCH")
from llmbench.vlmeval import run_vl_eval

spec = importlib.util.spec_from_file_location(
    "merge_vlmeval_into_card",
    "$LLMBENCH/scripts/merge_vlmeval_into_card.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

card = json.loads(Path("$CARD").read_text())
modalities = card.get("modalities") or "text+image"
vl = run_vl_eval(
    base_url="$BASE_URL",
    model="$SERVED_MODEL",
    tasks=$py_tasks,
    modalities=modalities,
    out_dir="$OUT",
    execute=None,
    smoke_endpoint=True,
)
print(json.dumps(vl.to_dict(), ensure_ascii=False, indent=2))
Path("$OUT/last_vl_result.json").write_text(
    json.dumps(vl.to_dict(), ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
)
mod.merge_from_vl_result(Path("$CARD"), vl, dry_run=False)
print("MERGED", "$CARD", "status=", vl.status)
PY
}

main(){
  : >"$MAIN"
  log "=== VLM DocVQA/OCR queue model=$MODEL_CARD ==="
  assert_safe
  run_eval
  log "DONE"
}

main "$@"
