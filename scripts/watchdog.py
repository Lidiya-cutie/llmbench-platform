#!/usr/bin/env python3
"""Watchdog: проверка критичных процессов. Алерт при падении."""
import os, sys, json, subprocess, datetime
from pathlib import Path

PROCS = {
    5997: "index_batch.py (S3)",
    1609537: "lm_eval qwen2.5-vl-7b mmlu_pro",
    2213730: "lm_eval gemma-3-27b-awq mmlu_pro",
    2243889: "lm_eval deepseek-v4-flash minerva_math",
    2212997: "gemma_triton_openai_proxy.py",
}

STATE = Path("out/.watchdog.json")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def main():
    now = datetime.datetime.now().isoformat(timespec="seconds")
    prev = json.loads(STATE.read_text()) if STATE.exists() else {"alive": {}, "alerted": []}
    alive_now = {}
    newly_dead = []
    still_alive = []

    for pid, name in PROCS.items():
        a = pid_alive(pid)
        alive_now[str(pid)] = a
        was = prev.get("alive", {}).get(str(pid), True)
        if not a and was:
            newly_dead.append(f"  PID {pid}: {name}")
        elif a:
            still_alive.append(f"  PID {pid}: {name}")

    STATE.write_text(json.dumps({"alive": alive_now, "ts": now}, indent=2))

    if newly_dead:
        print(f"[WATCHDOG {now}] FALLEN:")
        for d in newly_dead:
            print(d)
        print("Still alive:")
        for s in still_alive:
            print(s)
        sys.exit(1)
    elif still_alive:
        print(f"[WATCHDOG {now}] {len(still_alive)}/{len(PROCS)} alive")
    else:
        print(f"[WATCHDOG {now}] all down!")
        sys.exit(2)


if __name__ == "__main__":
    main()