"""
Аппаратный монитор для одной RTX 4090.

Семплирует питание/VRAM/утилизацию/температуру в фоновом потоке, пока идёт
оценка, и отдаёт агрегаты. Энергия считается трапецией по фактическим отсчётам
питания (а не power*time), поэтому корректно ловит всплески.

Приоритет источника метрик: pynvml -> nvidia-smi (subprocess) -> заглушка.

ВАЖНО про VRAM:
  vLLM резервирует VRAM заранее (gpu_memory_utilization), поэтому nvml покажет
  выделение, а не рабочий набор. Для llama.cpp (в т.ч. local-27b) VRAM отражает
  реальную загрузку весов + KV-cache и метрика достоверна. Это фиксируется в
  поле `vram_note`.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    import pynvml  # type: ignore
    _HAS_NVML = True
except Exception:  # pragma: no cover - зависит от окружения
    _HAS_NVML = False


@dataclass
class HwSample:
    t: float          # монотонное время, с
    power_w: float    # мгновенная мощность платы, Вт
    vram_mb: float    # занятая VRAM, МиБ
    util_pct: float   # утилизация SM, %
    temp_c: float     # температура ядра, °C


@dataclass
class HwSummary:
    samples: int
    duration_s: float
    energy_wh: float          # интеграл мощности по времени
    avg_power_w: float
    max_power_w: float
    peak_vram_mb: float
    avg_util_pct: float
    max_temp_c: float
    source: str               # 'pynvml' | 'nvidia-smi' | 'stub'
    vram_note: str = ""
    raw: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)  # сырьё не тащим в карточку
        return d


def _trapz_energy_wh(samples: List[HwSample]) -> float:
    if len(samples) < 2:
        return 0.0
    joules = 0.0
    for a, b in zip(samples, samples[1:]):
        dt = b.t - a.t
        joules += 0.5 * (a.power_w + b.power_w) * dt  # Вт*с = Дж
    return joules / 3600.0  # Дж -> Вт*ч


class _NvmlBackend:
    source = "pynvml"

    def __init__(self, index: int = 0):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)

    def read(self) -> HwSample:
        power = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0  # мВт -> Вт
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.h)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.h)
        temp = pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)
        return HwSample(time.monotonic(), power, mem.used / (1024 ** 2), float(util.gpu), float(temp))

    def close(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


class _SmiBackend:
    source = "nvidia-smi"
    _Q = ("power.draw", "memory.used", "utilization.gpu", "temperature.gpu")

    def __init__(self, index: int = 0):
        self.index = index

    def read(self) -> HwSample:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={self.index}",
             "--query-gpu=" + ",".join(self._Q),
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip().splitlines()[0]
        p, m, u, t = [x.strip() for x in out.split(",")]
        return HwSample(time.monotonic(), float(p), float(m), float(u), float(t))

    def close(self):
        pass


class _StubBackend:
    source = "stub"

    def __init__(self, index: int = 0):
        self._n = 0

    def read(self) -> HwSample:
        self._n += 1
        return HwSample(time.monotonic(), 0.0, 0.0, 0.0, 0.0)

    def close(self):
        pass


def _pick_backend(index: int):
    if _HAS_NVML:
        try:
            return _NvmlBackend(index)
        except Exception:
            pass
    try:
        subprocess.check_output(["nvidia-smi", "-L"], timeout=5)
        return _SmiBackend(index)
    except Exception:
        return _StubBackend(index)


class HardwareMonitor:
    """Контекст-менеджер: with HardwareMonitor() as hw: ...; hw.summary()."""

    def __init__(self, index: int = 0, interval_s: float = 0.1, engine_hint: str = ""):
        self.index = index
        self.interval_s = interval_s
        self.engine_hint = engine_hint.lower()
        self._backend = None
        self._samples: List[HwSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._t1 = 0.0

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._samples.append(self._backend.read())
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "HardwareMonitor":
        self._backend = _pick_backend(self.index)
        self._samples.clear()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._t1 = time.monotonic()
        if self._backend:
            self._backend.close()

    def summary(self) -> HwSummary:
        s = self._samples
        dur = (self._t1 or time.monotonic()) - self._t0
        note = ""
        if "vllm" in self.engine_hint or "sglang" in self.engine_hint:
            note = ("VRAM у vLLM/SGLang преаллоцирована (gpu_memory_utilization); "
                    "peak_vram_mb отражает резерв, а не рабочий набор.")
        elif "llama" in self.engine_hint or "ollama" in self.engine_hint:
            note = "VRAM для llama.cpp/ollama отражает фактическую загрузку весов + KV-cache."
        if not s:
            return HwSummary(0, dur, 0, 0, 0, 0, 0, 0,
                             getattr(self._backend, "source", "stub"), note, [])
        return HwSummary(
            samples=len(s),
            duration_s=round(dur, 3),
            energy_wh=round(_trapz_energy_wh(s), 6),
            avg_power_w=round(sum(x.power_w for x in s) / len(s), 2),
            max_power_w=round(max(x.power_w for x in s), 2),
            peak_vram_mb=round(max(x.vram_mb for x in s), 1),
            avg_util_pct=round(sum(x.util_pct for x in s) / len(s), 1),
            max_temp_c=round(max(x.temp_c for x in s), 1),
            source=self._backend.source,
            vram_note=note,
            raw=[asdict(x) for x in s],
        )
