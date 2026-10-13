#!/usr/bin/env python3
"""
Гетерогенные hardware-профили для llmbench.

Позволяет указывать GPU тип (A100, H100, RTX 4090, CPU, Apple Silicon)
и получать корректные cost-параметры, bandwidth и ограничения.

Ничего не трогает в существующих модулях.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareProfile:
    name: str
    gpu_type: str
    vram_gb: float
    memory_bw_gb_s: float
    fp16_tflops: float
    gpu_price_usd: float
    gpu_life_hours: float
    electricity_price_kwh: float
    pue: float
    max_power_w: float
    supports_tensor_parallel: bool
    max_gpus_for_tp: int
    notes: str = ""


# ---- Предопределённые профили ----

PROFILES: dict[str, HardwareProfile] = {
    "rtx4090": HardwareProfile(
        name="NVIDIA RTX 4090",
        gpu_type="ada_lovelace",
        vram_gb=24.0,
        memory_bw_gb_s=1008.0,
        fp16_tflops=82.6,
        gpu_price_usd=1800.0,
        gpu_life_hours=26280,
        electricity_price_kwh=0.15,
        pue=1.2,
        max_power_w=450.0,
        supports_tensor_parallel=False,
        max_gpus_for_tp=1,
        notes="Домашняя карта. Не поддерживает TP (1 GPU max).",
    ),
    "a100_40gb": HardwareProfile(
        name="NVIDIA A100 (40GB)",
        gpu_type="ampere",
        vram_gb=40.0,
        memory_bw_gb_s=1555.0,
        fp16_tflops=312.0,
        gpu_price_usd=15000.0,
        gpu_life_hours=52560,
        electricity_price_kwh=0.12,
        pue=1.1,
        max_power_w=400.0,
        supports_tensor_parallel=True,
        max_gpus_for_tp=8,
        notes="Датацентр. TP до 8 карт. MIG-поддержка.",
    ),
    "a100_80gb": HardwareProfile(
        name="NVIDIA A100 (80GB)",
        gpu_type="ampere",
        vram_gb=80.0,
        memory_bw_gb_s=2039.0,
        fp16_tflops=312.0,
        gpu_price_usd=20000.0,
        gpu_life_hours=52560,
        electricity_price_kwh=0.12,
        pue=1.1,
        max_power_w=400.0,
        supports_tensor_parallel=True,
        max_gpus_for_tp=8,
        notes="80GB HBM2e. Крупные модели.",
    ),
    "h100": HardwareProfile(
        name="NVIDIA H100 (80GB)",
        gpu_type="hopper",
        vram_gb=80.0,
        memory_bw_gb_s=3352.0,
        fp16_tflops=989.0,
        gpu_price_usd=35000.0,
        gpu_life_hours=52560,
        electricity_price_kwh=0.12,
        pue=1.1,
        max_power_w=700.0,
        supports_tensor_parallel=True,
        max_gpus_for_tp=8,
        notes="Hopper. FP8 трансформер-движок. TP до 8.",
    ),
    "h200": HardwareProfile(
        name="NVIDIA H200 (141GB)",
        gpu_type="hopper",
        vram_gb=141.0,
        memory_bw_gb_s=4800.0,
        fp16_tflops=989.0,
        gpu_price_usd=40000.0,
        gpu_life_hours=52560,
        electricity_price_kwh=0.12,
        pue=1.1,
        max_power_w=700.0,
        supports_tensor_parallel=True,
        max_gpus_for_tp=8,
        notes="H200 с HBM3e. Для 70B+ моделей.",
    ),
    "cpu_generic": HardwareProfile(
        name="CPU (generic)",
        gpu_type="cpu",
        vram_gb=0.0,
        memory_bw_gb_s=50.0,
        fp16_tflops=0.0,
        gpu_price_usd=0.0,
        gpu_life_hours=0,
        electricity_price_kwh=0.15,
        pue=1.0,
        max_power_w=0.0,
        supports_tensor_parallel=False,
        max_gpus_for_tp=0,
        notes="CPU-only. Используйте GGUF с offload=0.",
    ),
    "apple_m1_max": HardwareProfile(
        name="Apple M1 Max (64GB)",
        gpu_type="apple_silicon",
        vram_gb=64.0,
        memory_bw_gb_s=400.0,
        fp16_tflops=10.4,
        gpu_price_usd=3500.0,
        gpu_life_hours=35040,
        electricity_price_kwh=0.15,
        pue=1.0,
        max_power_w=60.0,
        supports_tensor_parallel=False,
        max_gpus_for_tp=0,
        notes="Unified memory, MLX/llama.cpp. Не CUDA.",
    ),
    "apple_m2_ultra": HardwareProfile(
        name="Apple M2 Ultra (192GB)",
        gpu_type="apple_silicon",
        vram_gb=192.0,
        memory_bw_gb_s=800.0,
        fp16_tflops=27.0,
        gpu_price_usd=7000.0,
        gpu_life_hours=35040,
        electricity_price_kwh=0.15,
        pue=1.0,
        max_power_w=120.0,
        supports_tensor_parallel=False,
        max_gpus_for_tp=0,
        notes="192GB unified. Крупные кванты.",
    ),
}


def get_profile(name: str) -> HardwareProfile:
    """Возвращает профиль по имени (регистронезависимо)."""
    name_lower = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    for key, profile in PROFILES.items():
        if key.lower().replace("-", "") == name_lower:
            return profile
    raise KeyError(f"Hardware profile '{name}' не найден. "
                   f"Доступны: {', '.join(PROFILES.keys())}")


def list_profiles() -> list[str]:
    return list(PROFILES.keys())


def cost_for_profile(profile: HardwareProfile, power_w: float) -> dict:
    """
    Рассчитывает стоимость для профиля при заданной мощности.
    Возвращает: {usd_per_hour, usd_per_1m_tokens}
    """
    gpu_amort = profile.gpu_price_usd / max(profile.gpu_life_hours, 1)
    power_kw = (power_w / 1000.0) * profile.pue
    energy_cost = power_kw * profile.electricity_price_kwh
    return {
        "usd_per_hour": round(gpu_amort + energy_cost, 5),
        "amortization_per_hour": round(gpu_amort, 6),
        "energy_per_hour": round(energy_cost, 5),
        "profile": profile.name,
    }
