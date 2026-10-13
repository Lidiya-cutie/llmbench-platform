#!/usr/bin/env python3
"""Offline rescore mbpp/mbpp_instruct samples (strip [DONE]/ fences + code_eval)."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")


def clean_code(resp: str) -> str:
    text = (resp or "").replace("\r", "")
    text = text.replace("[DONE]", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fences = [
        m.group(1).strip()
        for m in re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.S | re.I)
    ]
    if fences:
        for code in fences:
            if "def " in code:
                return code
        return fences[-1]
    m = re.search(r"(def\s+\w+[\s\S]*)", text)
    return (m.group(1).strip() if m else text.strip())


def _raw_resp(row: dict) -> str:
    # Prefer raw model output: extract_code filter often leaves empty filtered_resps.
    resps = row.get("resps") or [[""]]
    try:
        raw = str(resps[0][0] or "")
        if raw.strip():
            return raw
    except Exception:
        pass
    fr = row.get("filtered_resps")
    if isinstance(fr, list) and fr:
        x = fr[0]
        if isinstance(x, list):
            x = x[0] if x else ""
        return str(x or "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--task-key", default="mbpp")
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
        code = clean_code(_raw_resp(r))
        if not code.strip():
            empty += 1
        preds.append([code])
        refs.append(r.get("target") or (r.get("doc") or {}).get("test_list") or "")
        if isinstance(refs[-1], list):
            refs[-1] = "\n".join(str(x) for x in refs[-1])

    metrics, details = code_eval.compute(references=refs, predictions=preds, k=[1])
    pass1 = float(metrics["pass@1"])
    n_pass = sum(1 for v in details.values() if v and v[0][1].get("passed"))

    out_dir = args.out_dir or sp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out = out_dir / f"results_rescored_{stamp}.json"
    key = args.task_key
    payload = {
        "results": {
            key: {
                "alias": key,
                "pass_at_1,none": pass1,
                "pass_at_1,extract_code": pass1,
                "sample_len": len(rows),
                "rescored_offline": True,
                "method": "strip_DONE_fence_code_eval",
                "empty_extract": empty,
                "n_pass": n_pass,
            }
        },
        "source_samples": str(sp),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pass@1={pass1:.6f} n={len(rows)} n_pass={n_pass} empty={empty} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
