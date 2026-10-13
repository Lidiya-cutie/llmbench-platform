#!/usr/bin/env python3
"""Wrapper: запускает evaluate_model.py с checkpoint/resume.

При обрыве и рестарте — завершённые шаги пропускаются.
Checkpoint хранится в out/.ckpt_<model_key>.json

Usage:
  python scripts/run_with_ckpt.py --model-key MODEL --base-url URL ...
  (те же аргументы, что evaluate_model.py, плюс --resume)
"""
from __future__ import annotations
import argparse, json, os, sys, subprocess
from pathlib import Path


def ckpt_path(model_key, out_dir="out"):
    return Path(out_dir) / f".ckpt_{model_key}.json"


def load_ckpt(model_key, out_dir="out"):
    p = ckpt_path(model_key, out_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {"done": [], "model_key": model_key}


def save_step(model_key, step, out_dir="out"):
    p = ckpt_path(model_key, out_dir)
    c = load_ckpt(model_key, out_dir)
    if step not in c["done"]:
        c["done"].append(step)
    p.write_text(json.dumps(c, indent=2))
    print(f"[ckpt] saved: {step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--served-model", default=None)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--perf-n", type=int, default=8)
    ap.add_argument("--perf-max-tokens", type=int, default=256)
    ap.add_argument("--context-scaling", action="store_true")
    ap.add_argument("--avg-tokens-per-request", type=float, default=400.0)
    ap.add_argument("--no-thinking-body", action="store_true")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--resume", action="store_true", help="skip completed steps")
    args = ap.parse_args()

    ckpt = load_ckpt(args.model_key, args.out_dir)
    done = ckpt.get("done", []) if args.resume else []

    # build command for evaluate_model.py
    cmd = [sys.executable, "scripts/evaluate_model.py",
           "--model-key", args.model_key,
           "--config", args.config,
           "--out-dir", args.out_dir,
           "--perf-n", str(args.perf_n),
           "--perf-max-tokens", str(args.perf_max_tokens),
           "--avg-tokens-per-request", str(args.avg_tokens_per_request)]

    if args.base_url:
        cmd += ["--base-url", args.base_url]
    if args.served_model:
        cmd += ["--served-model", args.served_model]
    if args.suite:
        cmd += ["--suite", args.suite]
    if args.context_scaling:
        cmd += ["--context-scaling"]
    if args.no_thinking_body:
        cmd += ["--no-thinking-body"]

    # Step 1: full evaluate_model.py run
    # evaluate_model does health + micro + perf + lm_eval + card in one shot.
    # We wrap it: if it crashes, we save nothing.
    # If it succeeds, we save all steps.
    if not done:
        print(f"[wrapper] running evaluate_model.py for {args.model_key}...")
        ret = subprocess.run(cmd, cwd=os.getcwd())
        if ret.returncode == 0:
            for step in ["health", "micro", "perf", "lm_eval", "cost", "card"]:
                save_step(args.model_key, step, args.out_dir)
            print(f"[wrapper] all steps done, checkpoint saved")
            return 0
        else:
            print(f"[wrapper] evaluate_model.py failed (rc={ret.returncode})")
            print(f"[wrapper] no checkpoint saved — fix issue and re-run with --resume")
            return ret.returncode
    else:
        print(f"[wrapper] already completed: {done}")
        print(f"[wrapper] nothing to do (use without --resume to force full rerun)")
        return 0


if __name__ == "__main__":
    sys.exit(main())