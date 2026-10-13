#!/usr/bin/env python3
"""Сводная сравнительная таблица из out/card_*.json."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

QUALITY_COLS = (
    "gsm8k",
    "minerva_math",
    "mmlu_pro",
    "bbh",
    "humaneval",
    "humaneval_plus",
    "mbpp_plus",
    "gpqa_main_zeroshot",
)
PERF_COLS = (
    ("gen_tok_s_mean", "gen_tok/s"),
    ("ttft_s_mean", "TTFT,s"),
    ("prefill_tok_s_mean", "prefill_tok/s"),
    ("latency_s_p50", "lat_p50,s"),
)
HW_COLS = (
    ("peak_vram_mb", "VRAM_MiB"),
    ("avg_power_w", "power_W"),
    ("avg_util_pct", "util_%"),
)
COST_COLS = (
    ("usd_per_1m_tokens", "$/1M_tok"),
    ("usd_per_hour", "$/h"),
)
MICRO_COLS = (
    ("math_acc", "micro_math"),
    ("json_valid", "micro_json"),
    ("tool_call_ok", "micro_tool"),
)


def _get(d: dict, *path: str, default: Any = None) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p, default)
    return cur


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, bool):
        return "ok" if v else "no"
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 10:
            return f"{v:.2f}"
        return f"{v:.{nd}f}"
    return str(v)


def _delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "—"
    d = a - b
    return f"{d:+.3f}"


def _load_cards(indir: str, keys: Optional[List[str]]) -> List[dict]:
    cards: List[dict] = []
    for name in sorted(os.listdir(indir)):
        if not (name.startswith("card_") and name.endswith(".json")):
            continue
        key = name[len("card_") : -len(".json")]
        if keys and key not in keys:
            continue
        path = os.path.join(indir, name)
        with open(path, encoding="utf-8") as f:
            card = json.load(f)
        card["_key"] = key
        cards.append(card)
    return cards


def _baseline(cards: List[dict], baseline: Optional[str]) -> Optional[dict]:
    if not baseline:
        return None
    for c in cards:
        if c.get("_key") == baseline:
            return c
    return None


def build_compare(cards: List[dict], baseline: Optional[str]) -> dict:
    base = _baseline(cards, baseline)
    base_key = base["_key"] if base else None
    rows: List[dict] = []
    for c in cards:
        q = c.get("quality") or {}
        perf = c.get("perf") or {}
        hw = c.get("hardware") or {}
        cost = c.get("cost") or {}
        micro = c.get("micro") or {}
        q_filtered = {k: q.get(k) for k in QUALITY_COLS if k in q}
        q_meta = c.get("quality_meta") or {}
        if not q_filtered:
            q_status = "empty"
        elif q_meta.get("partial") or "mmlu_pro" not in q_filtered:
            q_status = "partial"
        else:
            q_status = "full"
        row = {
            "model": c["_key"],
            "engine": c.get("engine", ""),
            "quant": c.get("quant_format", ""),
            "generated_at": c.get("generated_at", ""),
            "quality": q_filtered,
            "quality_status": q_status,
            "micro": {k: micro.get(k) for k, _ in MICRO_COLS},
            "perf": {
                k: perf.get(k)
                for k, _ in PERF_COLS
                if perf.get(k) is not None
            },
            "hardware": {
                k: hw.get(k) for k, _ in HW_COLS if hw.get(k) is not None
            },
            "cost": {
                k: cost.get(k)
                for k, _
                in COST_COLS
                if cost.get(k) is not None
            },
        }
        if base and c["_key"] != base_key:
            bq = base.get("quality") or {}
            bp = base.get("perf") or {}
            bh = base.get("hardware") or {}
            row["delta_vs_baseline"] = {
                "gsm8k": _delta(q.get("gsm8k"), bq.get("gsm8k")),
                "mmlu_pro": _delta(q.get("mmlu_pro"), bq.get("mmlu_pro")),
                "gen_tok_s": _delta(
                    perf.get("gen_tok_s_mean"), bp.get("gen_tok_s_mean")
                ),
                "vram_mb": _delta(
                    hw.get("peak_vram_mb"), bh.get("peak_vram_mb")
                ),
            }
        rows.append(row)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline": base_key,
        "models": [c["_key"] for c in cards],
        "rows": rows,
    }


def calculate_ranking_score(card: dict, baseline_card: dict) -> float:
    """Расчет комплексного рейтинга модели."""
    score = 0.0
    perf = card.get("perf") or {}
    gen_speed = perf.get("gen_tok_s_mean")
    if gen_speed:
        baseline_speed = (baseline_card.get("perf") or {}).get(
            "gen_tok_s_mean", 1
        )
        if baseline_speed > 0:
            score += 40 * (gen_speed / baseline_speed - 0.5)

    hw = card.get("hardware") or {}
    vram = hw.get("peak_vram_mb")
    if vram and 4096 <= vram <= 24576:
        score += 30 * (1 - (vram - 4096) / 20480)

    cost = card.get("cost") or {}
    tokens_cost = cost.get("usd_per_1m_tokens")
    if tokens_cost:
        baseline_cost = (baseline_card.get("cost") or {}).get(
            "usd_per_1m_tokens", 1000
        )
        if baseline_cost > 0:
            score += 20 * (baseline_cost / tokens_cost - 0.5)

    quality = card.get("quality") or {}
    if quality:
        gsm8k = quality.get("gsm8k", 0) or 0
        score += 10 * gsm8k

    return max(0.0, min(100.0, score))


def render_markdown(data: dict) -> str:
    lines = [
        "# Сравнение моделей (llmbench)",
        "",
        f"_Сгенерировано: {data['generated_at']}_",
        f"_Baseline: `{data.get('baseline') or '—'}`_",
        f"_Моделей: {len(data['models'])}_",
        "",
    ]
    models = data["rows"]
    if not models:
        lines.append("_Нет карточек в out/_")
        return "\n".join(lines)

    quality_present = sorted(
        {k for r in models for k in (r.get("quality") or {})}
    )
    headers = ["Модель", "Движок", "Квант", "Дата", "Quality_Статус"]
    headers.extend(label for _, label in MICRO_COLS)
    headers.extend(quality_present)
    headers.extend(label for _, label in PERF_COLS)
    headers.extend(label for _, label in HW_COLS)
    headers.extend(label for _, label in COST_COLS)

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    def _status_label(r: dict) -> str:
        st = r.get("quality_status") or (
            "full" if r.get("quality") else "empty"
        )
        return {
            "full": "full (lm-eval)",
            "partial": "partial (gsm8k/minerva, no mmlu_pro)",
            "empty": "empty (lm-eval not done)",
        }.get(st, st)

    for r in models:
        q = r.get("quality") or {}
        status = _status_label(r)
        micro = r.get("micro") or {}
        perf = r.get("perf") or {}
        hw = r.get("hardware") or {}
        cost = r.get("cost") or {}
        cells = [
            r["model"],
            r.get("engine") or "—",
            (r.get("quant") or "—")[:20],
            (r.get("generated_at") or "")[:10] or "—",
            status,
        ]
        cells.extend(_fmt(micro.get(k)) for k, _ in MICRO_COLS)
        cells.extend(_fmt(q.get(k)) for k in quality_present)
        cells.extend(_fmt(perf.get(k)) for k, _ in PERF_COLS)
        cells.extend(_fmt(hw.get(k)) for k, _ in HW_COLS)
        cells.extend(_fmt(cost.get(k)) for k, _ in COST_COLS)
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Качество данных", ""])
    lines.append("| Модель | Статус качества |")
    lines.append("| --- | --- |")
    for r in models:
        lines.append(f"| {r['model']} | {_status_label(r)} |")

    if data.get("ranking"):
        lines.extend(["", "## Рейтинг", ""])
        lines.append("| Модель | Рейтинг | Базовый |")
        lines.append("| --- | --- | --- |")
        for item in data["ranking"]:
            lines.append(
                f"| {item['model']} | {item['score']} | {item.get('baseline', '—')} |"
            )

    lines.extend(["", "## Карточки", ""])
    for r in models:
        lines.append(f"- `{r['model']}` — {r.get('generated_at') or '—'}")

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Сводка card_*.json в compare.md/json")
    p.add_argument("--indir", default="out")
    p.add_argument("--out-md", default="out/compare.md")
    p.add_argument("--out-json", default="out/compare.json")
    p.add_argument(
        "--baseline",
        default="qwen3-8b-bf16",
        help="ключ модели-baseline для дельт",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="только эти ключи (по умолчанию все card_*.json)",
    )
    p.add_argument(
        "--ranking",
        action="store_true",
        help="добавить комплексный рейтинг",
    )
    args = p.parse_args()

    cards = _load_cards(args.indir, args.models)
    if not cards:
        print(f"[!] Нет card_*.json в {args.indir}")
        raise SystemExit(1)

    data = build_compare(cards, args.baseline)
    base = _baseline(cards, args.baseline)
    if args.ranking and base:
        ranking = []
        for c in cards:
            ranking.append(
                {
                    "model": c["_key"],
                    "score": round(calculate_ranking_score(c, base), 2),
                    "baseline": base["_key"],
                }
            )
        ranking.sort(key=lambda x: x["score"], reverse=True)
        data["ranking"] = ranking

    out_md_dir = os.path.dirname(args.out_md)
    if out_md_dir:
        os.makedirs(out_md_dir, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(data))

    out_json_dir = os.path.dirname(args.out_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    empty = sum(1 for r in data["rows"] if not r.get("quality"))
    print(f"Моделей: {len(data['models'])}")
    print(f"  Baseline: {data.get('baseline')}")
    print(f"  Quality статус: empty={empty}/{len(data['rows'])}")
    print(f"  -> {args.out_md}")
    print(f"  -> {args.out_json}")


if __name__ == "__main__":
    raise SystemExit(main())
