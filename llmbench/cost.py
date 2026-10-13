"""
Модель стоимости эксплуатации локальной модели на 4090.

Себестоимость = амортизация GPU (за время работы) + электроэнергия (по факту
измеренного питания). Всё считается из реальных метрик hwmon + perf, без магии.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class CostInputs:
    gpu_price: float = 1800.0          # цена карты, $ (RTX 4090 ~ретейл)
    gpu_life_hours: float = 3 * 365 * 24  # срок амортизации, ч (3 года 24/7)
    electricity_price_kwh: float = 0.15   # $/кВт*ч (замените на локальный тариф)
    pue: float = 1.2                    # накладные (охлаждение/БП), 1.0 если дома


@dataclass
class CostResult:
    usd_per_hour: float
    usd_per_1m_tokens: float
    usd_per_1k_requests: float
    energy_wh_per_1k_tokens: float
    assumptions: dict

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_cost(*, avg_power_w: float, gen_tok_s: float,
                  avg_tokens_per_request: float, inp: CostInputs = None) -> CostResult:
    inp = inp or CostInputs()
    gpu_amort_per_h = inp.gpu_price / max(inp.gpu_life_hours, 1)
    power_kw = (avg_power_w / 1000.0) * inp.pue
    energy_cost_per_h = power_kw * inp.electricity_price_kwh
    usd_per_hour = gpu_amort_per_h + energy_cost_per_h

    tokens_per_hour = max(gen_tok_s, 1e-6) * 3600.0
    usd_per_1m = usd_per_hour / (tokens_per_hour / 1_000_000)
    usd_per_1k_req = usd_per_1m * (avg_tokens_per_request / 1_000_000) * 1000
    wh_per_1k_tok = (avg_power_w / max(gen_tok_s, 1e-6)) * 1000 / 3600.0

    return CostResult(
        usd_per_hour=round(usd_per_hour, 5),
        usd_per_1m_tokens=round(usd_per_1m, 4),
        usd_per_1k_requests=round(usd_per_1k_req, 5),
        energy_wh_per_1k_tokens=round(wh_per_1k_tok, 3),
        assumptions=asdict(inp),
    )
