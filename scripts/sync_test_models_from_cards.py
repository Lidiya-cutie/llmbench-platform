#!/usr/bin/env python3
"""Sync out/dashboard_v2/test_models.json scores from card_*.json (no index.html edits)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/opt/llm-data/llmbench")
TEST_JSON = ROOT / "out/dashboard_v2/test_models.json"
CORE = ["mmlu_pro", "gsm8k", "minerva_math", "humaneval", "mbpp", "ifeval", "gpqa"]


def card_to_entry(card: dict) -> dict:
    q = card.get("quality") or {}
    scores = {k: (float(q[k]) if isinstance(q.get(k), (int, float)) else None) for k in CORE}
    filled = sum(1 for k in CORE if scores.get(k) is not None and scores[k] > 0)
    status = "ready" if filled == 7 else ("in_progress" if filled else "in_progress")
    return {
        "key": card.get("name"),
        "name": card.get("name"),
        "engine": f"{card.get('engine', 'vLLM')} (test)",
        "quant": card.get("quant_format") or "BF16",
        "gpu": "GPU0" if "internvl" in str(card.get("name")) else "GPU1",
        "role": "test",
        "scores": scores,
        "detail": {
            "disk_size_gb": card.get("disk_size_gb"),
            "modalities": card.get("modalities"),
            "vision": card.get("vision"),
        },
        "comparison": {
            "status": status,
            "core7_filled": filled,
            "gaps": [k for k in CORE if not (scores.get(k) and scores[k] > 0)],
        },
    }


def main() -> None:
    keys = []
    if TEST_JSON.exists():
        try:
            keys = [m.get("key") for m in json.loads(TEST_JSON.read_text()) if m.get("key")]
        except Exception:
            keys = []
    if not keys:
        keys = ["internvl3-5-8b", "janus-pro-7b"]

    out = []
    for key in keys:
        path = ROOT / f"out/card_{key}.json"
        if not path.exists():
            out.append(
                {
                    "key": key,
                    "name": key,
                    "engine": "vLLM (test)",
                    "quant": "BF16",
                    "gpu": "",
                    "role": "test",
                    "scores": {},
                    "detail": {},
                    "comparison": {"status": "in_progress"},
                }
            )
            continue
        out.append(card_to_entry(json.loads(path.read_text())))

    TEST_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
