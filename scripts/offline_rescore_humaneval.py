#!/usr/bin/env python3
"""Offline rescore humaneval_instruct samples (fence extract + code_eval)."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")


def extract_code(resp: str, entry: str) -> str:
    text = resp or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fences = [
        m.group(1).replace("\r", "")
        for m in re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.S | re.I)
    ]
    for code in fences:
        if entry and f"def {entry}" in code:
            return code.strip()
    for code in fences:
        if "def " in code:
            return code.strip()
    if fences:
        return fences[-1].strip()
    m = re.search(rf"(def\s+{re.escape(entry)}\b[\s\S]*)", text) if entry else None
    if m:
        return m.group(1).strip()
    m = re.search(r"(def\s+\w+[\s\S]*)", text)
    return (m.group(1).strip() if m else text.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    sp = args.samples
    rows = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"empty samples: {sp}")

    import evaluate

    code_eval = evaluate.load("code_eval")
    preds: list[list[str]] = []
    refs: list[str] = []
    empty = 0
    for r in rows:
        doc = r.get("doc") or {}
        entry = doc.get("entry_point") or ""
        raw = (r.get("resps") or [[""]])[0][0]
        code = extract_code(raw, entry)
        if not code.strip():
            empty += 1
        preds.append([code])
        refs.append(doc.get("test") or r.get("target") or "")

    metrics, _ = code_eval.compute(references=refs, predictions=preds, k=[1])
    pass1 = float(metrics["pass@1"])
    out_dir = args.out_dir or sp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out = out_dir / f"results_rescored_{stamp}.json"
    payload = {
        "results": {
            "humaneval_instruct": {
                "alias": "humaneval_instruct",
                "pass@1,create_test": pass1,
                "sample_len": len(rows),
                "rescored_offline": True,
                "method": "fence_prefer_def_entry",
                "empty_extract": empty,
            }
        },
        "source_samples": str(sp),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pass@1={pass1:.6f} n={len(rows)} empty={empty} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
