#!/usr/bin/env bash
# GPU1-only: local-27b-1bit GPQA + MBPP, then Qwen2.5-VL HumanEval (until=[]).
# Never touch GPU0 / Triton Gemma.
set -uo pipefail

LLMBENCH=/opt/llm-data/llmbench
VENV=$LLMBENCH/.venv/bin/python
LOGDIR=$LLMBENCH/logs
MAIN=$LOGDIR/gpu1_1bit_gpqa_mbpp_then_vl_he.log
LOCAL27B_PORT=8078
VL_PORT=8077
LOCAL27B_GGUF=/opt/llm-data/local-27b-demo/models/gguf/27B/local-27b-Q1_0.gguf
LOCAL27B_BIN=/opt/llm-data/local-27b-demo/bin/cuda/llama-server
LOCAL27B_MODEL_NAME=local-27b-Q1_0.gguf
LOCAL27B_OUT=$LLMBENCH/out/lmeval_local-27b-1bit_queue
VL_OUT=$LLMBENCH/out/lmeval_qwen2.5-vl-7b_queue/humaneval_instruct_until_empty
VL_NAME=Qwen2.5-VL-7B-Instruct
DOCKER_NAME=vllm-vl-he-gpu1
HF_CACHE=/home/lidiya/.cache/huggingface

mkdir -p "$LOGDIR" "$LOCAL27B_OUT/gpqa_generative" "$LOCAL27B_OUT/mbpp_instruct" "$VL_OUT"
: >"$MAIN"
log(){ echo "[$(date -Is)] $*" | tee -a "$MAIN"; }

gpu1_free_mib(){
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '
}
gpu0_busy_triton(){
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader -i 0 2>/dev/null | rg -qi triton
}
has_results(){ find "$1" -name 'results_20*.json' ! -name '*rescored*' 2>/dev/null | grep -q .; }

assert_safe(){
  if ! gpu0_busy_triton; then
    log "WARN: Triton not seen on GPU0 (continuing; still only using CUDA_VISIBLE_DEVICES=1 / docker device=1)"
  else
    log "OK: Triton present on GPU0 — leave it alone"
  fi
  local free
  free=$(gpu1_free_mib)
  log "GPU1 free=${free} MiB"
  if (( free < 18000 )); then
    log "ABORT: GPU1 not free enough ($free MiB)"
    exit 1
  fi
  if pgrep -af 'llama-server' | rg -v 'rg|gpu1_1bit' >/dev/null; then
    log "ABORT: unexpected llama-server already running:"
    pgrep -af llama-server | tee -a "$MAIN"
    exit 1
  fi
  if docker ps --format '{{.Names}}' | rg -qx 'vllm-bench|vllm-bench-queue|vllm-mmlu-pro|vllm-vl-he-gpu1'; then
    log "ABORT: conflicting vLLM container on host"
    docker ps --format '{{.Names}} {{.Ports}}' | tee -a "$MAIN"
    exit 1
  fi
}

stop_local-27b(){
  if [[ -n "${LOCAL27B_PID:-}" ]] && kill -0 "$LOCAL27B_PID" 2>/dev/null; then
    log "stop local-27b pid=$LOCAL27B_PID"
    kill "$LOCAL27B_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$LOCAL27B_PID" 2>/dev/null || true
  fi
  # only kill our port listener if still held by llama-server
  local p
  p=$(ss -ltnp 2>/dev/null | rg ":${LOCAL27B_PORT}\\b" | rg -o 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)
  if [[ -n "$p" ]] && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | rg -q llama-server; then
    log "stop leftover llama-server pid=$p on :$LOCAL27B_PORT"
    kill "$p" 2>/dev/null || true
    sleep 1
    kill -9 "$p" 2>/dev/null || true
  fi
}

start_local-27b(){
  stop_local-27b
  [[ -x "$LOCAL27B_BIN" ]] || { log "ERROR missing $LOCAL27B_BIN"; exit 1; }
  [[ -f "$LOCAL27B_GGUF" ]] || { log "ERROR missing $LOCAL27B_GGUF"; exit 1; }
  log "START local-27b-1bit on GPU1 :$LOCAL27B_PORT"
  # CRITICAL: llama.cpp splits ctx across --parallel slots.
  # ctx=4096 + parallel=2 => 2048/slot; GPQA~2790 / MBPP~4300 tokens => HTTP 400.
  # Use single slot + large ctx so long n-shot prompts fit.
  CUDA_VISIBLE_DEVICES=1 LOCAL27B_THINKING=0 \
    "$LOCAL27B_BIN" \
      --model "$LOCAL27B_GGUF" \
      --host 127.0.0.1 \
      --port "$LOCAL27B_PORT" \
      --ctx-size 16384 \
      --n-gpu-layers 999 \
      --parallel 1 \
      --reasoning off \
      >"$LOGDIR/local-27b_1bit_gpu1_gap.log" 2>&1 &
  LOCAL27B_PID=$!
  echo "$LOCAL27B_PID" >"$LOGDIR/local-27b_1bit_gpu1_gap.pid"
  log "local-27b pid=$LOCAL27B_PID"
  local i
  for i in $(seq 1 90); do
    if curl -sf -m 3 "http://127.0.0.1:${LOCAL27B_PORT}/health" >/dev/null 2>&1 \
       || curl -sf -m 3 "http://127.0.0.1:${LOCAL27B_PORT}/v1/models" >/dev/null 2>&1; then
      log "local-27b ready after ${i}s"
      # smoke
      curl -sf -m 30 "http://127.0.0.1:${LOCAL27B_PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${LOCAL27B_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2=\"}],\"max_tokens\":8}" \
        | tee -a "$MAIN" | head -c 300; echo | tee -a "$MAIN"
      return 0
    fi
    if ! kill -0 "$LOCAL27B_PID" 2>/dev/null; then
      log "ERROR local-27b died; tail log:"
      tail -40 "$LOGDIR/local-27b_1bit_gpu1_gap.log" | tee -a "$MAIN"
      exit 1
    fi
    sleep 2
  done
  log "ERROR local-27b health timeout"
  tail -40 "$LOGDIR/local-27b_1bit_gpu1_gap.log" | tee -a "$MAIN"
  exit 1
}

run_lm(){
  local name="$1" task="$2" out="$3" max_gen="$4" conc="$5" base="$6" model="$7" gen="${8:-}"
  mkdir -p "$out"
  if has_results "$out"; then
    log "SKIP $name (results exist in $out)"
    return 0
  fi
  log "START $name task=$task conc=$conc max_gen=$max_gen gen=${gen:-default}"
  cd "$LLMBENCH"
  export HF_ALLOW_CODE_EVAL=1
  local args=(
    -m lm_eval --model local-chat-completions
    --model_args "model=${model},base_url=${base}/v1/chat/completions,num_concurrent=${conc},max_length=16384,max_gen_toks=${max_gen},tokenized_requests=False"
    --tasks "$task" --apply_chat_template --batch_size 1
    --confirm_run_unsafe_code
    --output_path "$out" --log_samples
  )
  if [[ -n "$gen" ]]; then args+=(--gen_kwargs "$gen"); fi
  "$VENV" "${args[@]}" >"$LOGDIR/${name}.log" 2>&1
  local rc=$?
  if has_results "$out"; then
    log "OK $name rc=$rc"
  else
    log "FAIL $name rc=$rc (no results); tail:"
    tail -30 "$LOGDIR/${name}.log" | tee -a "$MAIN"
  fi
  return $rc
}

stop_vl(){
  docker rm -f "$DOCKER_NAME" >/dev/null 2>&1 || true
}

start_vl(){
  stop_vl
  log "START vLLM VL on GPU1 :$VL_PORT"
  docker run -d --name "$DOCKER_NAME" \
    --gpus '"device=1"' \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -p ${VL_PORT}:8077 \
    mlops-triton-vllm:24.12-qwen3 \
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
      --host 0.0.0.0 --port 8077 \
      --dtype bfloat16 --max-model-len 8192 \
      --gpu-memory-utilization 0.90 \
      --trust-remote-code \
      --served-model-name "$VL_NAME" \
    >>"$MAIN" 2>&1
  local i
  for i in $(seq 1 180); do
    if curl -sf -m 5 "http://127.0.0.1:${VL_PORT}/v1/models" 2>/dev/null | rg -q "$VL_NAME"; then
      log "VL ready after ${i} checks"
      return 0
    fi
    if [[ $((i % 15)) -eq 0 ]]; then
      log "wait VL... $i/180 status=$(docker inspect -f '{{.State.Status}}' "$DOCKER_NAME" 2>/dev/null || echo gone)"
    fi
    sleep 2
  done
  log "ERROR VL not ready"
  docker logs --tail 50 "$DOCKER_NAME" 2>&1 | tee -a "$MAIN"
  exit 1
}

merge_gaps(){
  log "MERGE cards"
  "$VENV" - <<'PY' >>"$MAIN" 2>&1
import json, re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/opt/llm-data/llmbench")

def latest_results(root: Path):
    files = sorted(root.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    files = [p for p in files if "rescored" not in p.name or True]
    return files[-1] if files else None

def pick_metric(res: dict, prefs):
    # prefs: list of substrings in priority
    nums = {k: v for k, v in res.items() if isinstance(v, (int, float)) and "stderr" not in k and k != "alias"}
    for pref in prefs:
        for k, v in nums.items():
            if pref in k:
                return float(v), k
    # fallback first numeric excluding sample_len
    for k, v in nums.items():
        if k != "sample_len":
            return float(v), k
    return None, None

def merge(card_path: Path, mappings):
    card = json.loads(card_path.read_text())
    q = card.setdefault("quality", {})
    meta = card.setdefault("quality_meta", {})
    changed = []
    for key, (root, prefs) in mappings.items():
        f = latest_results(root)
        if not f:
            continue
        d = json.loads(f.read_text())
        results = d.get("results") or {}
        # take first task blob
        blob = None
        for _, v in results.items():
            if isinstance(v, dict):
                blob = v
                break
        if not blob:
            continue
        val, mk = pick_metric(blob, prefs)
        if val is None:
            continue
        # pass@1 / exact_match already 0..1
        q[key] = val
        meta[key] = {"source": str(f.relative_to(ROOT)), "metric": mk, "updated": datetime.now(timezone.utc).isoformat()}
        changed.append((key, val, mk, f.name))
    # status
    core = ["mmlu_pro","gpqa","gsm8k","minerva_math","humaneval","mbpp","ifeval"]
    cov = sum(1 for k in core if float(q.get(k) or 0) > 0)
    cmp = card.setdefault("comparison", {})
    if cov >= 7:
        cmp["status"] = "ready"
    elif card_path.name.endswith("1bit.json"):
        cmp["status"] = "excluded" if cov < 7 else "ready"
    elif "vl" in card_path.name:
        cmp["status"] = "ready" if cov >= 7 else "in_progress"
    card["quality"] = q
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n")
    print(card_path.name, "cov", f"{cov}/7", "changed", changed, "status", cmp.get("status"))

# 1bit
merge(
    ROOT / "out/card_local-27b-1bit.json",
    {
        "gpqa": (ROOT / "out/lmeval_local-27b-1bit_queue/gpqa_generative", ["flexible-extract", "exact_match", "acc"]),
        "mbpp": (ROOT / "out/lmeval_local-27b-1bit_queue/mbpp_instruct", ["pass@1", "pass_at_1"]),
        "humaneval": (ROOT / "out/lmeval_local-27b-1bit_queue/humaneval_instruct", ["pass@1", "pass_at_1"]),
    },
)
# VL HE
merge(
    ROOT / "out/card_qwen2.5-vl-7b.json",
    {
        "humaneval": (ROOT / "out/lmeval_qwen2.5-vl-7b_queue/humaneval_instruct_until_empty", ["pass@1", "pass_at_1"]),
    },
)
print("merge done", datetime.now(timezone.utc).isoformat())
PY
}

cleanup(){
  log "cleanup trap"
  stop_local-27b
  stop_vl
}
trap cleanup EXIT

log "=== GPU1 gap queue: 1bit GPQA+MBPP → VL HumanEval ==="
assert_safe

# --- Phase A: local-27b 1bit ---
start_local-27b
run_lm 1bit_gpqa gpqa_main_generative_n_shot \
  "$LOCAL27B_OUT/gpqa_generative" 256 2 \
  "http://127.0.0.1:${LOCAL27B_PORT}" "$LOCAL27B_MODEL_NAME" || log "gpqa failed (continue to mbpp)"
run_lm 1bit_mbpp mbpp_instruct \
  "$LOCAL27B_OUT/mbpp_instruct" 512 2 \
  "http://127.0.0.1:${LOCAL27B_PORT}" "$LOCAL27B_MODEL_NAME" || {
    log "mbpp_instruct failed; try plain mbpp"
    run_lm 1bit_mbpp_plain mbpp \
      "$LOCAL27B_OUT/mbpp" 512 2 \
      "http://127.0.0.1:${LOCAL27B_PORT}" "$LOCAL27B_MODEL_NAME" || true
  }
stop_local-27b
# free VRAM pause
sleep 5
free=$(gpu1_free_mib); log "GPU1 free after local-27b stop: ${free} MiB"

# --- Phase B: VL HE ---
start_vl
run_lm vl_he humaneval_instruct \
  "$VL_OUT" 1024 2 \
  "http://127.0.0.1:${VL_PORT}" "$VL_NAME" 'until=[]' || true
stop_vl

merge_gaps
trap - EXIT
log "=== ALL DONE ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$MAIN"
