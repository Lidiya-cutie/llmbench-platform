"""Tests for llmbench.lmeval metric parsing."""
import json
from pathlib import Path

from llmbench.lmeval import parse_results_file, _pick_primary


def test_pick_primary_gsm8k():
    """gsm8k: exact_match,strict-match should be picked."""
    metrics = {
        "exact_match,strict-match": 0.3336,
        "exact_match_stderr,strict-match": 0.013,
        "exact_match,flexible-extract": 0.5739,
        "sample_len": 1319,
    }
    assert _pick_primary(metrics) == 0.3336


def test_pick_primary_minerva_math():
    """minerva_math: math_verify,none should be picked, not sample_len."""
    metrics = {
        "exact_match,none": 0.0022,
        "math_verify,none": 0.2418,
        "sample_len": 5000,
    }
    assert _pick_primary(metrics) == 0.2418


def test_pick_primary_mmlu_pro():
    """mmlu_pro: exact_match,custom-extract should be picked, not sample_len."""
    metrics = {
        "exact_match,custom-extract": 0.4517,
        "exact_match_stderr,custom-extract": 0.0044,
        "sample_len": 12032,
    }
    assert _pick_primary(metrics) == 0.4517


def test_pick_primary_humaneval():
    """humaneval: pass@1,create_test should be picked."""
    metrics = {
        "pass@1,create_test": 0.0,
        "pass@1_stderr,create_test": 0.0,
        "sample_len": 164,
    }
    assert _pick_primary(metrics) == 0.0


def test_pick_primary_fallback_no_match():
    """If no preferred key matches, pick first non-stderr numeric."""
    metrics = {
        "weird_metric,none": 0.5,
        "weird_stderr,none": 0.1,
    }
    assert _pick_primary(metrics) == 0.5


def test_pick_primary_empty():
    assert _pick_primary({}) == 0.0
