#!/usr/bin/env python3
import json, glob, os, sys
from pathlib import Path
from collections import defaultdict

OUT = Path(os.environ.get("LLMBENCH_OUT", "/opt/llm-data/llmbench/out"))

MODEL_MAP = {
    "Ternary-local-27b-Q2_0.gguf": "local-27b-27b-ternary",
    "zai-org__glm-4-9b-chat": "glm-4-9b-chat",
    "Qwen2.5-VL-7B-Instruct": "qwen2.5-vl-7b",
    "Qwen3-8B": "qwen3-8b-bf16",
    "qwen3__8b": "ollama-qwen3-8b",
    "deepseek-v4-flash-dspark": "deepseek-v4-flash",
    "gemma-3-27b-awq": "gemma-3-27b-awq",
}

TASK_ORDER = ["mmlu_pro", "gsm8k", "minerva_math", "humaneval", "mbpp", "ifeval", "gpqa", "arc_challenge", "hellaswag", "truthfulqa_mc", "winogrande"]

results = defaultdict(dict)

for f in sorted(OUT.rglob("results_*.json")):
    try:
        data = json.loads(f.read_text())
    except Exception:
        continue
    cfg = data.get("config", {})
    margs = cfg.get("model_args", "")
    if isinstance(margs, dict):
        model_id = margs.get("model", "")
    else:
        model_id = cfg.get("model_name", "")
        for part in str(margs).split(","):
            if part.strip().startswith("model="):
                model_id = part.strip().split("=", 1)[1]
    model_key = MODEL_MAP.get(model_id, model_id)
    for task, metrics in data.get("results", {}).items():
        if task == "all":
            continue
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            if "stderr" in k or k in ("alias", "samples", "sample_len"):
                continue
            metric_name = k.split(",")[0] if "," in k else k
            tag = k.split(",")[1] if "," in k else ""
            label = f"{task} ({metric_name}"
            if tag:
                label += f", {tag}"
            label += ")"
            old = results[model_key].get(label)
            if old is None or v > old:
                results[model_key][label] = v

models = sorted(results.keys(), key=lambda m: TASK_ORDER)
tasks_seen = set()
for m in models:
    tasks_seen.update(k.split(" (")[0] for k in results[m])
tasks_sorted = [t for t in TASK_ORDER if t in tasks_seen] + sorted(tasks_seen - set(TASK_ORDER))

print(f"{'Model':<25}", end="")
for t in tasks_sorted:
    print(f" {t:>16}", end="")
print()
print("-" * (25 + 17 * len(tasks_sorted)))

for m in models:
    print(f"{m:<25}", end="")
    for t in tasks_sorted:
        vals = {k: v for k, v in results[m].items() if k.startswith(t + " (")}
        if not vals:
            print(f" {'—':>16}", end="")
        elif len(vals) == 1:
            print(f" {list(vals.values())[0]:>16.1%}", end="")
        else:
            best = max(vals.values())
            print(f" {best:>16.1%}", end="")
    print()

print()
print("Детализация по метрикам:")
for m in models:
    print(f"\n=== {m} ===")
    for k in sorted(results[m]):
        print(f"  {k}: {results[m][k]:.4f}")
