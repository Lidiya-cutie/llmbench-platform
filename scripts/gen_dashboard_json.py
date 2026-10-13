#!/usr/bin/env python3
import json, os
from pathlib import Path
from collections import defaultdict

OUT = Path(os.environ.get("LLMBENCH_OUT", "/opt/llm-data/llmbench/out"))

MODEL_META = {
    "local-27b-1bit": {"maker": "llama.cpp (PrismML fork) . GGUF Q1_0 (1.125 bpw)", "ev": "v", "context_max": 262144},
    "local-27b-27b-ternary": {"maker": "llama.cpp (PrismML fork) . GGUF Q2_0 (ternary, 1.58 bpw)", "ev": "v", "context_max": 262144},
    "deepseek-v4-flash": {"maker": "vLLM (remote OpenAI-compatible) . served (remote vLLM)", "ev": "v", "context_max": 500000},
    "gemma-3-27b-awq": {"maker": "Triton vLLM . AWQ", "ev": "e", "context_max": 8192},
    "glm-4-9b-chat": {"maker": "vLLM . BF16", "ev": "v", "context_max": 8192},
    "ollama-qwen3-8b": {"maker": "Ollama . GGUF Q4_K_M", "ev": "v", "context_max": 32768},
    "qwen2.5-vl-7b": {"maker": "vLLM . BF16", "ev": "v", "context_max": 8192},
    "qwen3-8b-bf16": {"maker": "vLLM . BF16", "ev": "v", "context_max": 8192},
}

MODEL_MAP = {
    "Ternary-local-27b-Q2_0.gguf": "local-27b-27b-ternary",
    "Ternary-local-27b-Q1_0.gguf": "local-27b-1bit",
    "zai-org__glm-4-9b-chat": "glm-4-9b-chat",
    "zai-org/glm-4-9b-chat": "glm-4-9b-chat",
    "Qwen2.5-VL-7B-Instruct": "qwen2.5-vl-7b",
    "Qwen3-8B": "qwen3-8b-bf16",
    "qwen3__8b": "ollama-qwen3-8b",
    "qwen3:8b": "ollama-qwen3-8b",
    "deepseek-v4-flash-dspark": "deepseek-v4-flash",
    "gemma-3-27b-awq": "gemma-3-27b-awq",
}

MAX_CTX = 262144
results = defaultdict(lambda: defaultdict(dict))

for f in sorted(OUT.rglob("results_*.json")):
    try:
        data = json.loads(f.read_text())
    except Exception:
        continue
    cfg = data.get("config", {})
    margs = cfg.get("model_args", "")
    model_id = cfg.get("model_name", "")
    if isinstance(margs, dict):
        model_id = margs.get("model", model_id)
    else:
        for part in str(margs).split(","):
            if part.strip().startswith("model="):
                model_id = part.strip().split("=", 1)[1]
    model_key = MODEL_MAP.get(model_id, model_id)

    # Фильтр: пропускаем result-файлы с малым sample_len (тестовые/битые запуски)
    min_sample_map = {"gsm8k": 100, "mmlu_pro": 100, "minerva_math": 100,
                      "humaneval": 50, "mbpp": 50, "ifeval": 50, "gpqa": 50}

    for task, metrics in data.get("results", {}).items():
        if task == "all":
            continue
        # Проверяем sample_len — пропускаем тестовые запуски
        task_n = metrics.get("sample_len", 0)
        min_n = min_sample_map.get(task, 0)
        if isinstance(task_n, (int, float)) and task_n < min_n and task_n > 0:
            continue
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            if "stderr" in k or k in ("alias", "samples", "sample_len"):
                continue
            tag = k.split(",")[1] if "," in k else ""
            metric_base = k.split(",")[0]
            label = f"{metric_base}|{tag}" if tag else metric_base
            old = results[model_key][task].get(label)
            if old is None or v > old:
                results[model_key][task][label] = v

models_out = []
for name in MODEL_META:
    r = results.get(name, {})
    meta = MODEL_META[name]

    # Предпочитаемые метрики для каждого бенчмарка (приоритетный порядок)
    PREFERRED_METRICS = {
        "gsm8k": ["exact_match|flexible-extract", "exact_match|strict-match", "exact_match|none"],
        "minerva_math": ["math_verify|none", "exact_match|none"],
        "mmlu_pro": ["exact_match|custom-extract", "exact_match|none", "acc|none", "acc"],
        "humaneval": ["pass@1|create_test", "pass_at_1|none", "pass@1"],
        "mbpp": ["pass_at_1|none", "pass@1"],
        "ifeval": ["prompt_level_loose_acc|none", "inst_level_loose_acc|none", "prompt_level_strict_acc|none"],
        "gpqa": ["acc|none", "exact_match|none"],
    }

    def best(task, default=0):
        if task not in r:
            return default
        metrics = r[task]
        if not metrics:
            return default
        # Ищем по предпочтительному списку
        preferred = PREFERRED_METRICS.get(task, [])
        for pref_key in preferred:
            if pref_key in metrics:
                v = metrics[pref_key]
                return round(v * 100, 1) if v <= 1.0 else round(v, 1)
        # Fallback: max из всех
        v = max(metrics.values())
        return round(v * 100, 1) if v <= 1.0 else round(v, 1)

    qcats_mmlu = {}
    qcats_minerva = {}
    for task_key, metrics_dict in r.items():
        if task_key.startswith("mmlu_pro_") and task_key != "mmlu_pro":
            v = max(metrics_dict.values()) if metrics_dict else 0
            qcats_mmlu[task_key] = round(v * 100, 1) if v <= 1.0 else round(v, 1)
        elif task_key.startswith("minerva_math_") and task_key != "minerva_math":
            v = max(metrics_dict.values()) if metrics_dict else 0
            qcats_minerva[task_key] = round(v * 100, 1) if v <= 1.0 else round(v, 1)

    gsm8k = best("gsm8k")
    minerva = best("minerva_math")
    math_max = max(gsm8k, minerva) if max(gsm8k, minerva) > 0 else 0
    ctx_norm = min(100, round(meta["context_max"] / MAX_CTX * 100, 1))

    models_out.append({
        "name": name,
        "maker": meta["maker"],
        "ev": meta["ev"],
        "quality": {
            "mmlu_pro": best("mmlu_pro"),
            "minerva_math": minerva,
            "gsm8k": gsm8k,
            "humaneval": best("humaneval"),
            "mbpp": best("mbpp"),
            "ifeval": best("ifeval"),
            "gpqa": best("gpqa"),
            "math_max": math_max
        },
        "quality_categories": {"mmlu_pro": qcats_mmlu, "minerva_math": qcats_minerva},
        "perf": {"gen_tok_s_mean": 0, "ttft_s_mean": 0, "gen_tok_s_median": 0,
                 "prefill_tok_s_mean": 0, "ttft_s_p90": 0, "ttft_s_p95": 0,
                 "latency_s_p50": 0, "latency_s_p90": 0, "latency_s_p95": 0, "latency_s_p99": 0},
        "hardware": {"energy_wh": 0, "avg_power_w": 0, "max_power_w": 0,
                     "peak_vram_mb": 0, "avg_util_pct": 0, "max_temp_c": 0, "samples": 0},
        "cost": {"usd_per_hour": 0, "usd_per_1m_tokens": 0,
                 "usd_per_1k_requests": 0, "energy_wh_per_1k_tokens": 0},
        "context_max": meta["context_max"],
        "context_normalized": ctx_norm
    })

output = {"models": models_out, "generated_at": str(os.path.getmtime(__file__))}
OUT.joinpath("api_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
print(f"Written {len(models_out)} models to {OUT}/api_results.json")
