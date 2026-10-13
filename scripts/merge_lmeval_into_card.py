#!/usr/bin/env python3
"""Безопасное вливание частичных lm-eval результатов в card_*.json без GPU.

Не трогает serving / lm_eval процессы. Пишет backup рядом с карточкой.
По умолчанию не перезаписывает compare.json — только preview.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_TASK_METRIC_PREFS: Dict[str, Tuple[str, ...]] = {
    "gsm8k": ("exact_match,flexible-extract", "exact_match,strict-match"),
    "minerva_math": ("math_verify,none", "exact_match,none"),
    "mmlu_pro": ("exact_match,custom-extract", "acc,none", "acc"),
    "humaneval": ("pass@1,create_test", "pass_at_1,none", "pass@1"),
    "humaneval_instruct": ("pass@1,create_test", "pass_at_1,none", "pass@1"),
    "mbpp": ("pass_at_1,extract_coder", "pass@1", "pass_at_1,none"),
    "mbpp_instruct": ("pass_at_1,extract_coder", "pass@1", "pass_at_1,none"),
    "ifeval": (
        "prompt_level_loose_acc,none",
        "inst_level_loose_acc,none",
        "prompt_level_strict_acc,none",
    ),
    "gpqa": (
        "exact_match,flexible-extract",
        "exact_match,none",
        "acc,none",
        "acc",
    ),
    "gpqa_main_generative_n_shot": (
        "exact_match,flexible-extract",
        "exact_match,none",
        "acc,none",
        "acc",
    ),
}


def _task_dirs(task_dir: Path) -> list[Path]:
    """task_dir plus sibling variants: *_fixed, *_nothink, *_new, *instruct*."""
    parent = task_dir.parent
    name = task_dir.name
    dirs: list[Path] = []
    seen = set()
    if task_dir.is_dir():
        dirs.append(task_dir)
        seen.add(task_dir)
    if parent.is_dir():
        for d in parent.iterdir():
            if not d.is_dir() or d in seen:
                continue
            if d.name.startswith(name) or name.startswith(d.name.rstrip("_")):
                dirs.append(d)
                seen.add(d)
            elif any(
                x in d.name
                for x in ("_fixed", "_nothink", "_new", "instruct", name.split("_")[0])
            ) and name.split("_")[0] in d.name:
                dirs.append(d)
                seen.add(d)
    return dirs


def _score_result_file(path: Path, task_hint: str) -> Optional[Tuple[float, dict]]:
    """Primary score for ranking; prefers canonical metrics over any float."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    results = data.get("results") or {}
    prefs = _TASK_METRIC_PREFS.get(task_hint) or _TASK_METRIC_PREFS.get(
        task_hint.replace("_instruct", "")
    )
    best = None
    meta: dict = {"sample_len": data.get("n-samples") or data.get("sample_len")}
    for task_name, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue
        if prefs:
            for pref in prefs:
                if pref in task_metrics and isinstance(
                    task_metrics[pref], (int, float)
                ):
                    best = float(task_metrics[pref])
                    meta["metric"] = pref
                    meta["task"] = task_name
                    break
        if best is not None:
            break
        nums = []
        for k, v in task_metrics.items():
            if (
                isinstance(v, (int, float))
                and "stderr" not in k
                and k != "alias"
            ):
                nums.append(float(v))
        if nums:
            best = nums[0]
            meta["task"] = task_name
    if best is None:
        return None
    return best, meta


def _protocol_rank(path: Path) -> int:
    """Higher = preferred protocol source (nothink/recheck over silent best-score)."""
    s = str(path).lower()
    score = 0
    if "results_rescored" in path.name.lower() or "_rescored" in s:
        score += 50
    if "_recheck_nothink" in s:
        score += 30
    if "_nothink" in s:
        score += 20
    if "instruct" in s:
        score += 10
    if "until_empty" in s or "until=[]" in s:
        score += 15
    return score


def _latest_results(task_dir: Path) -> Optional[Tuple[Path, float, dict]]:
    """Pick results_*.json: prefer rescored/protocol dirs, then score."""
    candidates: list[Path] = []
    for d in _task_dirs(task_dir):
        candidates.extend(d.rglob("results_*.json"))
    scored = []
    task_hint = task_dir.name
    for f in candidates:
        got = _score_result_file(f, task_hint)
        if not got:
            continue
        score, meta = got
        scored.append((f, score, meta))
    if not scored:
        return None
    scored.sort(key=lambda t: (_protocol_rank(t[0]), t[1], t[0].stat().st_mtime))
    return scored[-1]


def extract_gsm8k(path: Path) -> Tuple[float, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = (data.get("results") or {}).get("gsm8k") or {}
    for key in ("exact_match,flexible-extract", "exact_match,strict-match"):
        if key in m:
            return float(m[key]), {"metric": key, "sample_len": m.get("sample_len")}
    raise KeyError(f"no gsm8k metric in {path}")


def extract_minerva_math(path: Path) -> Tuple[float, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    weighted = 0.0
    weight = 0.0
    subsets = {}
    for name, metrics in results.items():
        if not name.startswith("minerva_math"):
            continue
        n = float(metrics.get("sample_len") or metrics.get("alias") or 0) or 1.0
        v = None
        for mk in ("math_verify,none", "exact_match,none"):
            if mk in metrics and isinstance(metrics[mk], (int, float)):
                v = float(metrics[mk])
                break
        if v is None:
            continue
        subsets[name] = v
        weighted += v * n
        weight += n
    if weight <= 0:
        raise KeyError("no minerva_math")
    return weighted / weight, {
        "metric": "math_verify,none (weighted by sample_len)",
        "subsets": subsets,
        "sample_len": int(weight),
    }


def extract_mmlu_pro(path: Path) -> Tuple[float, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    m = results.get("mmlu_pro") or {}
    for key in ("exact_match,custom-extract", "acc,none", "acc"):
        if key in m and isinstance(m[key], (int, float)):
            return float(m[key]), {"metric": key, "sample_len": m.get("sample_len")}
    vals = []
    subsets = {}
    for name, metrics in results.items():
        if not name.startswith("mmlu_pro_"):
            continue
        for key in ("exact_match,custom-extract", "acc,none", "acc"):
            if key in metrics and isinstance(metrics[key], (int, float)):
                vals.append(float(metrics[key]))
                subsets[name] = float(metrics[key])
                break
    if not vals:
        raise KeyError(f"no mmlu_pro metric in {path}")
    return sum(vals) / len(vals), {"metric": "mean_subgroups", "subsets": subsets}


def extract_humaneval(path: Path) -> Tuple[float, dict]:
    """Извлечение humaneval / humaneval_instruct результатов (coding метрика)"""
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    for task_name, m in results.items():
        if "humaneval" not in task_name:
            continue
        for key in ("pass@1,create_test", "pass_at_1,none", "pass@1", "pass_at_1"):
            if key in m and isinstance(m[key], (int, float)):
                return float(m[key]), {
                    "metric": key,
                    "sample_len": m.get("sample_len"),
                    "task": task_name,
                }
    raise KeyError(f"no usable humaneval metric in {path}")


def extract_mbpp(path: Path) -> Tuple[float, dict]:
    """Извлечение mbpp / mbpp_instruct результатов (coding метрика)"""
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    for task_name, m in results.items():
        if "mbpp" not in task_name:
            continue
        for key in ("pass_at_1,extract_coder", "pass@1", "pass_at_1,none", "pass_at_1"):
            if key in m and isinstance(m[key], (int, float)):
                return float(m[key]), {
                    "metric": key,
                    "sample_len": m.get("sample_len"),
                    "task": task_name,
                }
    raise KeyError(f"no usable mbpp metric in {path}")


def extract_ifeval(path: Path) -> Tuple[float, dict]:
    """Извлечение ifeval результатов"""
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    m = results.get("ifeval") or next(iter(results.values()), {})
    for key in (
        "prompt_level_loose_acc,none",
        "inst_level_loose_acc,none",
        "prompt_level_strict_acc,none",
    ):
        if key in m and isinstance(m[key], (int, float)):
            return float(m[key]), {"metric": key, "sample_len": m.get("sample_len")}
    raise KeyError(f"no usable ifeval metric in {path}")


def extract_gpqa(path: Path) -> Tuple[float, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    for task_name, m in results.items():
        if not str(task_name).startswith("gpqa"):
            continue
        if not isinstance(m, dict):
            continue
        for key in (
            "exact_match,flexible-extract",
            "exact_match,none",
            "acc,none",
            "acc",
            "exact_match",
        ):
            if key in m and isinstance(m[key], (int, float)):
                return float(m[key]), {
                    "metric": key,
                    "sample_len": m.get("sample_len"),
                    "task": task_name,
                }
        # fallback: best non-stderr numeric (avoid strict-match=0 hiding flexible)
        nums = [
            (k, float(v))
            for k, v in m.items()
            if isinstance(v, (int, float))
            and "stderr" not in k
            and k not in ("alias", "sample_len")
        ]
        if nums:
            nums.sort(key=lambda t: (0 if "flexible" in t[0] else 1, -t[1]))
            k, v = nums[0]
            return v, {"metric": k, "sample_len": m.get("sample_len"), "task": task_name}
    raise KeyError(f"no usable gpqa metric in {path}")


def merge_into_card(
    card_path: Path, lmeval_root: Path, dry_run: bool = False
) -> dict:
    card = json.loads(card_path.read_text(encoding="utf-8"))
    existing_quality = dict(card.get("quality") or {})
    existing_perf = card.get("perf")
    existing_hardware = card.get("hardware")
    existing_micro = card.get("micro")
    existing_context_scaling = card.get("context_scaling")
    existing_cost = card.get("cost")
    existing_vision = card.get("vision")
    existing_reliability = card.get("reliability")
    existing_fit = card.get("fit")

    quality = dict(existing_quality)
    sources = dict((card.get("quality_meta") or {}).get("sources") or {})

    def _try(subdir: str, extract, key: str, extra: Optional[dict] = None):
        nonlocal quality, sources
        got = _latest_results(lmeval_root / subdir)
        if not got:
            return
        path, _score, _meta = got
        try:
            score, meta = extract(path)
        except Exception as e:
            sources[key] = {"error": str(e), "source": str(path)}
            return
        quality[key] = round(float(score), 6)
        entry = {"metric": meta.get("metric"), "sample_len": meta.get("sample_len"), "source": str(path)}
        if extra:
            entry.update(extra)
        if "task" in meta:
            entry["task_variant"] = meta["task"]
        sources[key] = entry

    _try("gsm8k", extract_gsm8k, "gsm8k")
    _try("minerva_math", extract_minerva_math, "minerva_math")
    _try("mmlu_pro", extract_mmlu_pro, "mmlu_pro")
    he = _latest_results(lmeval_root / "humaneval_instruct") or _latest_results(
        lmeval_root / "humaneval"
    )
    if he:
        path, _, _ = he
        try:
            score, meta = extract_humaneval(path)
            quality["humaneval"] = round(float(score), 6)
            sources["humaneval"] = {
                "metric": meta.get("metric"),
                "sample_len": meta.get("sample_len"),
                "source": str(path),
                "task_variant": meta.get("task"),
            }
        except Exception as e:
            sources["humaneval"] = {"error": str(e), "source": str(path)}

    mb = _latest_results(lmeval_root / "mbpp_instruct") or _latest_results(
        lmeval_root / "mbpp"
    )
    if mb:
        path, _, _ = mb
        try:
            score, meta = extract_mbpp(path)
            quality["mbpp"] = round(float(score), 6)
            sources["mbpp"] = {
                "metric": meta.get("metric"),
                "sample_len": meta.get("sample_len"),
                "source": str(path),
                "task_variant": meta.get("task"),
            }
        except Exception as e:
            sources["mbpp"] = {"error": str(e), "source": str(path)}

    _try("ifeval", extract_ifeval, "ifeval")
    gp = _latest_results(
        lmeval_root / "gpqa_main_generative_n_shot"
    ) or _latest_results(lmeval_root / "gpqa")
    if gp:
        path, _, _ = gp
        try:
            score, meta = extract_gpqa(path)
            quality["gpqa"] = round(float(score), 6)
            sources["gpqa"] = {
                "metric": meta.get("metric"),
                "sample_len": meta.get("sample_len"),
                "source": str(path),
                "task_variant": meta.get("task"),
            }
        except Exception as e:
            sources["gpqa"] = {"error": str(e), "source": str(path)}

    he_v = quality.get("humaneval") or 0
    mb_v = quality.get("mbpp") or 0
    if he_v or mb_v:
        quality["coding"] = max(float(he_v or 0), float(mb_v or 0))

    core = [
        "mmlu_pro",
        "gsm8k",
        "minerva_math",
        "humaneval",
        "mbpp",
        "ifeval",
        "gpqa",
    ]
    filled = sum(1 for k in core if float(quality.get(k) or 0) > 0)
    gaps = [k for k in core if float(quality.get(k) or 0) <= 0]
    note = (
        f"частичный lm-eval обновлен; сохранены perf/hardware/micro данные (perf_n="
        f"{(existing_perf or {}).get('n')}, ok={(existing_perf or {}).get('ok')})"
        if filled < 7
        else "полный набор quality из lm-eval; сохранены perf/hardware/micro данные"
    )

    card["quality"] = quality
    if existing_perf is not None:
        card["perf"] = existing_perf
    if existing_hardware is not None:
        card["hardware"] = existing_hardware
    if existing_micro is not None:
        card["micro"] = existing_micro
    if existing_context_scaling is not None:
        card["context_scaling"] = existing_context_scaling
    if existing_cost is not None:
        card["cost"] = existing_cost
    if existing_vision is not None:
        card["vision"] = existing_vision
    if existing_reliability is not None:
        card["reliability"] = existing_reliability
    if existing_fit is not None:
        card["fit"] = existing_fit

    card["quality_meta"] = {
        "partial": filled < 7,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
        "note": note,
    }
    card["comparison"] = {
        "status": "core7_partial" if filled < 7 else "core7_ready",
        "core7_filled": filled,
        "gaps": gaps,
        "note": note,
    }
    card["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not dry_run:
        bak = card_path.with_suffix(
            card_path.suffix
            + ".bak-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        shutil.copy2(card_path, bak)
        card_path.write_text(
            json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        bak_link = card_path.with_name(card_path.name + "_backup")
        try:
            if bak_link.exists() or bak_link.is_symlink():
                bak_link.unlink()
            bak_link.symlink_to(bak.name)
        except Exception:
            pass
    return card


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--card", default="out/card_glm-4-9b-chat.json")
    p.add_argument(
        "--lmeval-root",
        default="out/lmeval_glm_resumable",
        help="каталог с gsm8k/minerva_math/mmlu_pro",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--rebuild-compare-preview",
        action="store_true",
        help="после merge собрать out/compare_preview.{md,json} (не трогает compare.json)",
    )
    p.add_argument("--baseline", default="qwen3-8b-bf16")
    args = p.parse_args()

    card_path = Path(args.card)
    root = Path(args.lmeval_root)
    if not card_path.exists():
        print(f"[!] нет карточки {card_path}")
        raise SystemExit(1)
    card = merge_into_card(card_path, root, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "card": str(card_path),
                "quality": card.get("quality"),
                "comparison": card.get("comparison"),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.rebuild_compare_preview:
        try:
            from compare_models import _load_cards, build_compare, render_markdown
        except Exception:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from compare_models import _load_cards, build_compare, render_markdown

        cards = _load_cards(str(card_path.parent))
        data = build_compare(cards, baseline=args.baseline)
        out_dir = card_path.parent
        md_path = out_dir / "compare_preview.md"
        js_path = out_dir / "compare_preview.json"
        md_path.write_text(render_markdown(data), encoding="utf-8")
        js_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"preview -> {md_path} {js_path}")
        print("боевые out/compare.json и out/compare.md НЕ изменены")


if __name__ == "__main__":
    main()
