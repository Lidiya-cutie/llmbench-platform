"""Tests for llmbench.cost estimation."""
from llmbench.cost import CostInputs, estimate_cost


def test_estimate_cost_basic():
    inp = CostInputs(
        gpu_price=2000.0,
        gpu_life_hours=20000.0,
        electricity_price_kwh=5.0,
        pue=1.0,
    )
    cost = estimate_cost(
        avg_power_w=300.0,
        gen_tok_s=50.0,
        avg_tokens_per_request=400.0,
        inp=inp,
    )
    assert cost.usd_per_1m_tokens > 0
    assert cost.usd_per_hour > 0


def test_estimate_cost_zero_throughput():
    """Zero throughput should not crash."""
    inp = CostInputs(
        gpu_price=2000.0,
        gpu_life_hours=20000.0,
        electricity_price_kwh=5.0,
        pue=1.0,
    )
    cost = estimate_cost(
        avg_power_w=300.0,
        gen_tok_s=0.0,
        avg_tokens_per_request=400.0,
        inp=inp,
    )
    # Zero throughput → infinite or zero cost per token, but must not crash
    assert cost is not None
