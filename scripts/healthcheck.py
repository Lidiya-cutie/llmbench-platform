#!/usr/bin/env python3
"""
Healthcheck для системы llmbench.

Проверяет:
1. Python venv и ключевые модули
2. vLLM эндпоинт (HTTP /health)
3. GPU доступность (nvidia-smi)
4. Docker контейнеры
5. Карточки моделей
6. Дашборд файл
7. Прокси (опционально)

Exit codes: 0 = OK, 1 = WARN, 2 = CRITICAL
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

LLMBENCH = Path(os.environ.get("LLMBENCH_DIR", "/opt/llmbench"))
VENV_PY = LLMBENCH / ".venv" / "bin" / "python"
DASHBOARD = LLMBENCH / "out" / "dashboard_v2" / "index.html"
CARDS_DIR = LLMBENCH / "out"

VLLM_ENDPOINTS = [
    ("gpu0", os.environ.get("VLLM_PORT_GPU0", "8070")),
    ("gpu1", os.environ.get("VLLM_PORT_GPU1", "8077")),
]

CRITICAL = []
WARN = []


def check(name, ok, detail=""):
    status = "OK" if ok else "FAIL"
    line = f"  [{'✓' if ok else '✗'}] {name}: {status}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        CRITICAL.append(name)


def warn(name, ok, detail=""):
    status = "OK" if ok else "WARN"
    line = f"  [{'✓' if ok else '!'}] {name}: {status}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        WARN.append(name)


def run(cmd, timeout=10):
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode == 0, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def main():
    print("=== llmbench Healthcheck ===")
    print()

    # 1. Python venv
    print("[1/7] Python venv")
    check("venv exists", VENV_PY.exists(), str(VENV_PY))
    if VENV_PY.exists():
        ok, out = run(f"{VENV_PY} -c 'import llmbench; print(\"OK\")'", 10)
        check("llmbench import", ok, out)
        ok, out = run(f"{VENV_PY} -c 'import lm_eval; print(lm_eval.__version__)'", 10)
        check("lm_eval", ok, out)
        ok, out = run(
            f"LD_LIBRARY_PATH={VENV_PY.parent.parent}/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH "
            f"{VENV_PY} -c 'import vllm; print(vllm.__version__)'",
            15,
        )
        warn("vllm (needs LD_LIBRARY_PATH)", ok, out)
    print()

    # 2. vLLM endpoints
    print("[2/7] vLLM endpoints")
    for gpu_name, port in VLLM_ENDPOINTS:
        ok, out = run(f"curl -sf http://127.0.0.1:{port}/health", 5)
        warn(f"vllm-{gpu_name} :{port}", ok, out or "no response")
    print()

    # 3. GPU
    print("[3/7] GPU")
    ok, out = run("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader", 5)
    check("nvidia-smi", ok)
    if ok:
        for line in out.split("\n"):
            print(f"       {line}")
    print()

    # 4. Docker
    print("[4/7] Docker containers")
    ok, out = run("docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -E 'vllm|llmbench|triton'", 5)
    containers = out.split("\n") if out else []
    if containers and containers[0]:
        for c in containers:
            print(f"       {c}")
    else:
        warn("llmbench containers", False, "no vllm/llmbench containers running")
    print()

    # 5. Model cards
    print("[5/7] Model cards")
    cards = sorted(CARDS_DIR.glob("card_*.json"))
    check("cards found", len(cards) >= 7, f"{len(cards)} cards")
    for c in cards:
        print(f"       {c.name}")
    print()

    # 6. Dashboard
    print("[6/7] Dashboard")
    check("index.html exists", DASHBOARD.exists(), str(DASHBOARD))
    if DASHBOARD.exists():
        size = DASHBOARD.stat().st_size
        check("index.html > 10KB", size > 10240, f"{size // 1024}KB")
    bench_data = LLMBENCH / "out" / "dashboard_v2" / "bench_data.json"
    warn("bench_data.json exists", bench_data.exists())
    print()

    # 7. Proxy (optional)
    print("[7/7] Proxy (optional)")
    ok, out = run("curl -sf --socks5-hostname 127.0.0.1:10810 https://git.example.com --max-time 5 -o /dev/null -w '%{http_code}'", 10)
    warn("SOCKS5 proxy :10810", ok, out)
    print()

    # Summary
    print("=== Summary ===")
    if CRITICAL:
        print(f"  CRITICAL: {len(CRITICAL)} — {', '.join(CRITICAL)}")
    if WARN:
        print(f"  WARN: {len(WARN)} — {', '.join(WARN)}")
    if not CRITICAL and not WARN:
        print("  ALL OK")
        return 0
    elif CRITICAL:
        return 2
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
