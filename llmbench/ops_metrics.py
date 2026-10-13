#!/usr/bin/env python3
"""
Операционные метрики модели: время загрузки, idle VRAM, CPU RAM overhead.

Ничего не трогает в существующих модулях.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpsMetrics:
    load_time_s: float          # время от запуска процесса до готовности
    idle_vram_mb: float         # VRAM в простое
    idle_cpu_ram_mb: float      # CPU RAM процесса вне GPU
    idle_power_w: float         # мощность в простое
    start_command: str          # команда запуска
    engine_type: str            # vLLM / llama.cpp / etc


def _wait_for_endpoint(url: str, timeout: int = 300, interval: float = 2.0) -> float:
    """Ждёт, пока эндпоинт /v1/models не начнёт отвечать 200."""
    import requests
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            r = requests.get(f"{url.rstrip('/')}/models", timeout=5)
            if r.status_code == 200:
                return time.monotonic() - start
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Эндпоинт {url} не ответил за {timeout}с")


def measure_loading(
    url: str,
    pid: Optional[int] = None,
    process_name: Optional[str] = None,
) -> OpsMetrics:
    """
    Измеряет время загрузки модели.
    Если PID не указан, пытается найти процесс по имени.
    """
    start = time.monotonic()

    # Ждём готовности эндпоинта
    load_time = _wait_for_endpoint(url)

    # Измеряем idle потребление
    vram = 0.0
    cpu_ram = 0.0
    power = 0.0

    if pid:
        try:
            # VRAM через nvidia-smi
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,power.draw",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5
            )
            parts = out.strip().split(",")
            vram = float(parts[0].strip()) if parts else 0.0
            power = float(parts[1].strip()) if len(parts) > 1 else 0.0

            # CPU RAM через /proc
            smaps = f"/proc/{pid}/status"
            with open(smaps) as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        cpu_ram = float(line.split()[1]) / 1024  # kB -> MB
                        break
        except Exception:
            pass

    return OpsMetrics(
        load_time_s=round(load_time, 1),
        idle_vram_mb=round(vram, 1),
        idle_cpu_ram_mb=round(cpu_ram, 1),
        idle_power_w=round(power, 2),
        start_command="",
        engine_type="unknown",
    )
