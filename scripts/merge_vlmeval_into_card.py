#!/usr/bin/env python3
"""Вливание DocVQA/OCR (и др. vision) в card_*.json без поломки core-7.

Правила:
- scores пишутся в card.vision.* и quality_meta.sources
- в quality кладутся только реальные float (docvqa/ocrbench/...), без плейсхолдеров 0.0
- плейсхолдеры docvqa_score/ocrbench_score/vision_tested удаляются
- core-7 / comparison.core7_* не пересчитываются здесь
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmbench.modelcard import VISION_BENCHES, is_vision_capable  # noqa: E402
from llmbench.vlmeval import VlEvalResult, _normalize_primary  # noqa: E402


def _load_primary_from_dir(work_dir: Path) -> Dict[str, float]:
    from llmbench.vlmeval import _parse_kit_scores

    if not work_dir.exists():
        return {}
    return _parse_kit_scores(work_dir)


def _load_primary_json(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "primary" in data:
        raw = data["primary"]
    elif isinstance(data, dict):
        raw = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    else:
        raw = {}
    return _normalize_primary({str(k): float(v) for k, v in raw.items()})


def merge_vision_into_card(
    card_path: Path,
    *,
    primary: Optional[Dict[str, float]] = None,
    status: Optional[str] = None,
    note: Optional[str] = None,
    source: Optional[str] = None,
    dry_run: bool = False,
    strip_placeholders: bool = True,
) -> dict:
    card = json.loads(card_path.read_text(encoding="utf-8"))
    modalities = card.get("modalities") or "text"
    vision = dict(card.get("vision") or {})
    quality = dict(card.get("quality") or {})
    meta = dict(card.get("quality_meta") or {})
    sources = dict(meta.get("sources") or {})

    if strip_placeholders:
        for junk in (
            "docvqa_score",
            "ocrbench_score",
            "chartqa_score",
            "mmmu_score",
            "vision_tested",
            "vision_test_date",
        ):
            quality.pop(junk, None)

    if not is_vision_capable(modalities):
        vision = {
            "status": "not_tested",
            "note": "не тестировалось (text-only)",
            **{k: "not_tested" for k in VISION_BENCHES},
        }
        card["vision"] = vision
        card["quality"] = quality
        if not dry_run:
            _backup_write(card_path, card)
        return card

    primary = dict(primary or {})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if primary:
        for k, v in primary.items():
            if k not in VISION_BENCHES:
                continue
            vision[k] = round(float(v), 6)
            quality[k] = round(float(v), 6)
            sources[k] = {
                "metric": "vlmeval_primary",
                "source": source or "",
                "updated_at": now,
                "domain": "vision_ocr",
            }
        vision["status"] = status or "done"
        vision["note"] = note or f"vision scores: {primary}"
    else:
        vision.setdefault("status", status or vision.get("status") or "pending")
        if note:
            vision["note"] = note
        for k in VISION_BENCHES:
            if k not in vision:
                vision[k] = vision.get("status", "pending")

    vision["updated_at"] = now
    meta["sources"] = sources
    meta["vision_updated_at"] = now
    card["vision"] = vision
    card["quality"] = quality
    card["quality_meta"] = meta

    # coverage отдельного домена, без влияния на core7_filled
    filled = sum(1 for k in ("docvqa", "ocrbench") if isinstance(vision.get(k), (int, float)))
    card["vision_comparison"] = {
        "status": "vision_ready" if filled >= 2 else ("vision_partial" if filled else "vision_pending"),
        "docvqa_ocr_filled": filled,
        "gaps": [k for k in ("docvqa", "ocrbench") if not isinstance(vision.get(k), (int, float))],
        "benches": list(VISION_BENCHES),
    }

    if not dry_run:
        _backup_write(card_path, card)
    return card


def _backup_write(card_path: Path, card: dict) -> None:
    bak = card_path.with_suffix(
        card_path.suffix
        + ".bak-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    shutil.copy2(card_path, bak)
    card_path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def merge_from_vl_result(card_path: Path, vl: VlEvalResult, dry_run: bool = False) -> dict:
    return merge_vision_into_card(
        card_path,
        primary=vl.primary,
        status=vl.status,
        note=vl.note,
        source=vl.raw_path,
        dry_run=dry_run,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card", required=True)
    ap.add_argument("--work-dir", help="каталог kit_run / out/vlmeval_<model>/...")
    ap.add_argument("--scores-json", help="JSON с primary scores")
    ap.add_argument("--status")
    ap.add_argument("--note")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--fix-placeholders-only",
        action="store_true",
        help="только убрать fake docvqa_score=0 из quality + нормализовать vision",
    )
    args = ap.parse_args()

    card_path = Path(args.card)
    primary: Dict[str, float] = {}
    source = None
    if args.scores_json:
        source = args.scores_json
        primary = _load_primary_json(Path(args.scores_json))
    elif args.work_dir:
        source = args.work_dir
        primary = _load_primary_from_dir(Path(args.work_dir))

    if args.strip_placeholders_only:
        card = merge_vision_into_card(
            card_path,
            primary=None,
            status=args.status,
            note=args.note or "placeholders stripped; awaiting DocVQA/OCRBench",
            dry_run=args.dry_run,
            strip_placeholders=True,
        )
    else:
        card = merge_vision_into_card(
            card_path,
            primary=primary or None,
            status=args.status,
            note=args.note,
            source=source,
            dry_run=args.dry_run,
        )

    print(
        json.dumps(
            {
                "card": str(card_path),
                "vision": card.get("vision"),
                "vision_comparison": card.get("vision_comparison"),
                "quality_vision_keys": {
                    k: card.get("quality", {}).get(k) for k in VISION_BENCHES
                },
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
