#!/usr/bin/env python3
"""
Оркестратор: полная оценка ОДНОЙ модели по уже поднятому OpenAI-эндпоинту.

Предполагается, что сервер модели уже запущен (для local-27b — через демо-репо
PrismML: ./scripts/start_llama_server.sh на :8080). Этот скрипт НЕ поднимает
сервер сам, чтобы не завязываться на кастомные кернелы — он лишь бьёт по base_url.

Пример:
  python scripts/evaluate_model.py \\
      --model-key local-27b-1bit \\
      --base-url http://127.0.0.1:8077/v1 \\
      --served-model local-27b \\
      --suite smoke \\
      --avg-tokens-per-request 400

Что делает по шагам:
  1) micro-eval (математика/JSON/tool) + perf (TTFT/tok-s) под hwmon;
  2) (опц.) lm-eval-harness по выбранному suite под hwmon;
  3) считает стоимость из измеренных мощности и скорости;
  4) собирает карточку, пишет expert-заключение, сохраняет .md и .json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmbench import (HardwareMonitor, run_perf, run_context_scaling, run_micro_eval,
                      run_lm_eval, estimate_cost, CostInputs, ModelCard,
                      render_markdown, finalize, wait_ready, check_openai_endpoint)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--model-key", required=True, help="ключ из models.yaml")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--served-model", default=None, help="имя модели в /v1 (для llama.cpp часто 'local-27b')")
    ap.add_argument("--suite", default=None, help="имя suite из benchmark_suites; пропустить = без lm-eval")
    ap.add_argument("--perf-n", type=int, default=8)
    ap.add_argument("--perf-max-tokens", type=int, default=256)
    ap.add_argument("--context-scaling", action="store_true")
    ap.add_argument("--avg-tokens-per-request", type=float, default=400.0)
    ap.add_argument("--no-thinking-body", action="store_true",
                    help="добавить thinking_budget_tokens=0 в тело запроса (local-27b)")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--skip-health-check", action="store_true",
                    help="пропустить проверку готовности эндпоинта")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    m = cfg["models"][args.model_key]
    defaults = cfg.get("defaults", {})
    base_url = args.base_url or defaults.get("base_url", "http://127.0.0.1:8077/v1")
    served = args.served_model or m.get("served_model_name", args.model_key)
    hw_idx = defaults.get("hardware", {}).get("gpu_index", 0)
    interval = defaults.get("hardware", {}).get("sample_interval_s", 0.1)
    engine = m.get("engine", "")
    extra_body = {"thinking_budget_tokens": 0} if args.no_thinking_body else None

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[i] Оценка {args.model_key} @ {base_url} (model={served}, engine={engine})")

    # --- Health-check эндпоинта перед оценкой ---
    if not args.skip_health_check:
        print(f"[0/4] health-check эндпоинта {base_url} …")
        health = wait_ready(base_url, model_name=served, timeout=180, interval=5)
        if not health.ok:
            print(f"[✗] Эндпоинт недоступен: {health.error}")
            print(f"    models_list: {health.models_list}")
            sys.exit(1)
        print(f"      ✓ OK ({health.latency_ms}ms, {len(health.models_list)} models)")
        if served and served not in health.models_list:
            print(f"[!] Модель '{served}' не найдена в /v1/models: {health.models_list}")
            sys.exit(1)
    else:
        print("[0/4] health-check пропущен (--skip-health-check)")

    # --- Шаг 1: perf + micro под мониторингом железа ---
    with HardwareMonitor(index=hw_idx, interval_s=interval, engine_hint=engine) as hw:
        print("[1/4] micro-eval …")
        micro = run_micro_eval(base_url, served, extra_body=extra_body)
        print(f"      math={micro.math_acc}  json={micro.json_valid}  tool={micro.tool_call_ok}")
        print("[2/4] perf …")
        perf = run_perf(base_url, served, n=args.perf_n, max_tokens=args.perf_max_tokens,
                        extra_body=extra_body)
        ctx = []
        if args.context_scaling:
            print("      context scaling …")
            ctx = [r.to_dict() for r in run_context_scaling(base_url, served, extra_body=extra_body)]
    hws = hw.summary()
    print(f"      power~{hws.avg_power_w}W  VRAM~{hws.peak_vram_mb}MiB  gen~{perf.gen_tok_s_mean}tok/s")

    # --- Шаг 2: lm-eval (опционально) под мониторингом ---
    quality = {}
    if args.suite:
        suite = cfg["benchmark_suites"][args.suite]
        print(f"[3/4] lm-eval suite='{args.suite}' tasks={suite['tasks']} …")
        with HardwareMonitor(index=hw_idx, interval_s=interval, engine_hint=engine):
            le = run_lm_eval(base_url, served, suite["tasks"],
                             out_dir=os.path.join(args.out_dir, "lmeval"),
                             limit=suite.get("limit"))
        if le.ok:
            quality = le.primary
            print("      ", quality)
        else:
            print("      [!] lm-eval не отработал:", le.error)
    else:
        print("[3/4] lm-eval пропущен (--suite не задан)")

    # --- Шаг 3: стоимость ---
    ci = CostInputs(**defaults.get("cost", {}))
    cost = estimate_cost(avg_power_w=hws.avg_power_w or 350.0,
                         gen_tok_s=perf.gen_tok_s_mean or 1.0,
                         avg_tokens_per_request=args.avg_tokens_per_request, inp=ci)

    # --- Шаг 4: карточка + заключение ---
    print("[4/4] сборка карточки и заключения …")
    card = ModelCard(
        name=args.model_key, repo=m.get("repo", ""), base_model=m.get("base_model", ""),
        quant_format=m.get("quant_format", ""), disk_size_gb=m.get("disk_size_gb", 0.0),
        context_max=m.get("context_max", 0), modalities=m.get("modalities", "text"),
        license=m.get("license", ""), engine=engine, serving_note=m.get("launch_cmd", ""),
        thinking_mode=m.get("thinking_default", "n/a"),
        quality=quality, micro=micro.to_dict(), perf=perf.to_dict(), context_scaling=ctx,
        hardware=hws.to_dict(), cost=cost.to_dict(),
        vendor_claims=m.get("vendor_claims", []),
    )
    card = finalize(card)

    md_path = os.path.join(args.out_dir, f"card_{args.model_key}.md")
    js_path = os.path.join(args.out_dir, f"card_{args.model_key}.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(card))
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(card.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"\n✅ Готово:\n   {md_path}\n   {js_path}")
    print("\n--- Экспертное заключение ---\n" + card.conclusion)


if __name__ == "__main__":
    main()
