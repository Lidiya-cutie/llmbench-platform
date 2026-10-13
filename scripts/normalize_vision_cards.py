#!/usr/bin/env python3
"""Нормализация vision-полей во всех card_*.json без GPU.

- text-only → vision.status=not_tested, modalities остаются
- ложные text+image у text GGUF (local-27b) → modalities=text
- удаление fake quality.docvqa_score=0.0 плейсхолдеров
- core-7 не трогается
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# модели, у которых modalities ошибочно text+image (текстовый serving)
FORCE_TEXT = {
    "local-27b-1bit",
    "local-27b-27b-ternary",
}
VISION_KEYS = ("docvqa", "ocrbench", "chartqa", "mmmu")
JUNK = (
    "docvqa_score",
    "ocrbench_score",
    "chartqa_score",
    "mmmu_score",
    "vision_tested",
    "vision_test_date",
)


def normalize_card(path: Path, dry_run: bool = False) -> dict:
    card = json.loads(path.read_text(encoding="utf-8"))
    name = card.get("name") or path.stem.replace("card_", "")
    quality = dict(card.get("quality") or {})
    for j in JUNK:
        quality.pop(j, None)

    if name in FORCE_TEXT:
        card["modalities"] = "text"

    mods = (card.get("modalities") or "text").lower()
    vision = dict(card.get("vision") or {})
    if "image" not in mods and "vision" not in mods:
        vision = {
            "status": "not_tested",
            "note": "не тестировалось (text-only)",
            **{k: "not_tested" for k in VISION_KEYS},
        }
    else:
        vision.setdefault("status", "pending")
        vision.setdefault(
            "note",
            vision.get("note")
            or "VLM/OCR ожидает DocVQA_VAL + OCRBench (scripts/run_vlm_docvqa_ocr.sh)",
        )
        for k in VISION_KEYS:
            if k not in vision or vision[k] in (0, 0.0, "0", None):
                # не оставляем ложный 0.0 как «прогон»
                if not isinstance(vision.get(k), float) or vision.get(k) == 0.0:
                    if vision.get("status") == "done" and isinstance(vision.get(k), float):
                        pass
                    else:
                        vision[k] = vision.get("status", "pending")
        # убрать нули из quality для vision, если не done
        for k in VISION_KEYS:
            if k in quality and quality[k] in (0, 0.0) and vision.get("status") != "done":
                quality.pop(k, None)

    card["quality"] = quality
    card["vision"] = vision
    filled = sum(1 for k in ("docvqa", "ocrbench") if isinstance(vision.get(k), (int, float)))
    card["vision_comparison"] = {
        "status": "vision_ready"
        if filled >= 2
        else ("vision_partial" if filled else ("vision_na" if "image" not in mods else "vision_pending")),
        "docvqa_ocr_filled": filled,
        "gaps": [
            k
            for k in ("docvqa", "ocrbench")
            if "image" in mods and not isinstance(vision.get(k), (int, float))
        ],
    }

    if not dry_run:
        bak = path.with_suffix(
            path.suffix + ".bak-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        shutil.copy2(path, bak)
        path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "name": name,
        "modalities": card.get("modalities"),
        "vision_status": vision.get("status"),
        "vision_comparison": card["vision_comparison"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "out"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = []
    for path in sorted(Path(args.out_dir).glob("card_*.json")):
        rows.append(normalize_card(path, dry_run=args.dry_run))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
