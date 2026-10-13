"""Tests for llmbench.modelcard structure and rendering."""
import json

from llmbench.modelcard import ModelCard, assess_fit, build_conclusion, finalize


def _dummy_card(**overrides):
    defaults = dict(
        name="test-model",
        repo="test/repo",
        base_model="test-base",
        quant_format="BF16",
        disk_size_gb=10.0,
        context_max=8192,
        modalities="text",
        license="Apache-2.0",
        engine="vLLM",
        serving_note="",
        thinking_mode="n/a",
        quality={"gsm8k": 0.33, "mmlu_pro": 0.45},
        micro={"math_acc": 0.8, "json_valid": 1.0, "tool_call_ok": 0.0},
        perf={"gen_tok_s_mean": 42.0, "ttft_mean": 0.03, "prefill_tok_s_mean": 800.0,
              "latency_p50": 6.0, "latency_p95": 8.0},
        context_scaling=[],
        hardware={"avg_power_w": 280.0, "peak_vram_mb": 24000, "avg_gpu_util": 95.0},
        cost={"cost_per_1m_tokens": 0.77, "cost_per_hour": 0.12},
        vendor_claims=[],
    )
    defaults.update(overrides)
    return ModelCard(**defaults)


def test_card_to_dict_roundtrip():
    card = _dummy_card()
    d = card.to_dict()
    assert d["name"] == "test-model"
    assert d["quality"]["gsm8k"] == 0.33


def test_assess_fit():
    card = _dummy_card()
    fit = assess_fit(card)
    assert isinstance(fit, dict)
    assert len(fit) > 0


def test_build_conclusion():
    card = _dummy_card()
    conclusion = build_conclusion(card)
    assert isinstance(conclusion, str)
    assert len(conclusion) > 10


def test_finalize():
    card = _dummy_card()
    finalized = finalize(card)
    assert finalized.conclusion
    assert finalized.fit  # field is 'fit', not 'fit_verdict'


def test_card_with_empty_quality():
    card = _dummy_card(quality={})
    fit = assess_fit(card)
    assert isinstance(fit, dict)
