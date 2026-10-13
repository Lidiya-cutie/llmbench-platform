"""
VLM/OCR через OpenAI-совместимый image_url + опционально VLMEvalKit.

Безопасность:
- НЕ ставит пакеты в активный .venv llmbench (там может идти lm_eval).
- Полный DocVQA/OCRBench запускается только при LLMBENCH_VL_EXECUTE=1
  (.venv-vlmeval). Иначе — adapter_ready / not_implemented.
"""
from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .modelcard import VISION_BENCHES, is_vision_capable

ROOT = Path(__file__).resolve().parents[1]
VLMEVAL_KIT = ROOT / "third_party" / "VLMEvalKit"
DEFAULT_VL_PYTHON = ROOT / ".venv-vlmeval" / "bin" / "python"

TASK_TO_DATASET = {
    "docvqa": "DocVQA_VAL",
    "ocrbench": "OCRBench",
    "chartqa": "ChartQA_TEST",
    "mmmu": "MMMU_DEV_VAL",
}
DATASET_TO_TASK = {v: k for k, v in TASK_TO_DATASET.items()}
DATASET_TO_TASK.update(
    {
        "DocVQA": "docvqa",
        "OCRBench": "ocrbench",
        "OCRBench_MINI": "ocrbench",
        "ChartQA": "chartqa",
        "MMMU": "mmmu",
    }
)


@dataclass
class VlEvalResult:
    primary: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    note: str = ""
    ok: bool = False
    error: str = ""
    raw_path: str = ""

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "status": self.status,
            "note": self.note,
            "ok": self.ok,
            "error": self.error,
            "raw_path": self.raw_path,
        }

    def vision_block(self) -> dict:
        block = {
            "status": self.status or "not_tested",
            "note": self.note
            or (
                "не тестировалось (text-only)"
                if self.status == "skipped_text_only"
                else ""
            ),
        }
        for name in VISION_BENCHES:
            if name in self.primary and isinstance(self.primary[name], (int, float)):
                block[name] = float(self.primary[name])
            else:
                block[name] = self.status if self.status else "pending"
        return block


def chat_completions_url(base_url: str) -> str:
    u = (base_url or "").rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return u + "/chat/completions"
    return u + "/v1/chat/completions"


def normalize_tasks(tasks: List[str] | None) -> List[str]:
    out: List[str] = []
    for t in tasks or []:
        key = t.lower().replace("-", "").replace("_", "")
        for name in VISION_BENCHES:
            if key == name.replace("_", ""):
                out.append(name)
                break
    return out or ["docvqa", "ocrbench"]


def vlmeval_python(py: Optional[str] = None) -> Optional[str]:
    env = (os.environ.get("LLMBENCH_VLMEVAL_PYTHON") or "").strip()
    if env and Path(env).is_file():
        return env
    if py and Path(py).is_file():
        return py
    if DEFAULT_VL_PYTHON.is_file():
        return str(DEFAULT_VL_PYTHON)
    return None


def kit_importable(py: Optional[str] = None) -> Tuple[bool, str]:
    """Проверка vlmeval в отдельном интерпретаторе (не в активном .venv)."""
    exe = vlmeval_python(py)
    if not exe:
        return False, "нет .venv-vlmeval (scripts/setup_vlmeval_venv.sh)"
    try:
        r = subprocess.run(
            [exe, "-c", "import vlmeval; print('ok')"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(VLMEVAL_KIT) if VLMEVAL_KIT.is_dir() else None,
        )
        if r.returncode == 0 and "ok" in (r.stdout or ""):
            return True, "vlmeval import ok"
        return False, (r.stderr or r.stdout or "import failed").strip()[:400]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def openai_image_url_chat(
    base_url: str,
    model: str,
    prompt: str = "What is written in the image? Answer with one short word.",
    timeout: float = 60.0,
    api_key: str = "sk-local",
) -> Tuple[bool, str]:
    """
    Минимальный smoke адаптера image_url (1×1 PNG → chat/completions).
    Не тянет VLMEvalKit и не качает датасеты.
    """
    try:
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5XZ1sAAAAASUVORK5CYII="
        )
        b64 = base64.b64encode(png).decode("ascii")
        url = chat_completions_url(base_url)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
        }
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code}: {r.text[:300]}"
        data = r.json()
        content = (
            ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            or ""
        )
        return True, str(content)[:200]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _write_kit_config(path: Path, model_name: str, datasets: List[str]) -> None:
    data: dict = {}
    for ds in datasets:
        if ds == "OCRBench" or ds.startswith("OCRBench"):
            data[ds] = {"class": "OCRBench", "dataset": ds}
        elif ds.startswith("DocVQA"):
            data[ds] = {"class": "ImageVQADataset", "dataset": ds}
        elif ds.startswith("ChartQA"):
            data[ds] = {"class": "ImageVQADataset", "dataset": ds}
        elif ds.startswith("MMMU"):
            data[ds] = {"class": "MMMUDataset", "dataset": ds}
        else:
            data[ds] = {"dataset": ds}

    cfg = {
        "model": {
            "llmbench_local_vl": {
                "class": "GPT4V",
                "model": model_name,
                "temperature": 0,
                "img_size": 512,
                "img_detail": "low",
                "retry": 3,
                "verbose": False,
            }
        },
        "data": data,
    }
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_kit_scores(work_dir: Path) -> Dict[str, float]:
    """Грубый разбор score-файлов VLMEvalKit (csv/json/xlsx рядом с work_dir)."""
    primary: Dict[str, float] = {}
    score_keys = ("Overall", "overall", "accuracy", "acc", "Final Score", "Score")
    for p in work_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        task = None
        for ds, t in DATASET_TO_TASK.items():
            if ds.lower() in name:
                task = t
                break
        if task is None:
            continue

        if p.suffix == ".json":
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                for key in score_keys:
                    v = data.get(key)
                    if isinstance(v, (int, float)):
                        primary[task] = float(v)
                        break
                if task not in primary:
                    for k, v in data.items():
                        if isinstance(v, (int, float)) and "stderr" not in str(k).lower():
                            primary[task] = float(v)
                            break
        elif p.suffix == ".csv":
            try:
                with p.open(newline="", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
            except Exception:
                continue
            if len(rows) < 2:
                continue
            header = [h.strip() for h in rows[0]]
            for row in rows[1:]:
                for i, h in enumerate(header):
                    if h.lower() in ("overall", "final score", "total", "accuracy", "acc"):
                        try:
                            primary[task] = float(row[i])
                        except Exception:
                            pass
    return _normalize_primary(primary)


def _normalize_primary(primary: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in primary.items():
        x = float(v)
        if k == "ocrbench" and x > 1.5:
            x = x / 1000.0
        elif k in ("docvqa", "chartqa", "mmmu") and x > 1.5:
            x = x / 100.0
        out[k] = x
    return out


def run_vlmevalkit(
    base_url: str,
    model: str,
    tasks: List[str],
    out_dir: str = "out/vlmeval",
    limit: Optional[int] = None,
) -> VlEvalResult:
    py = vlmeval_python()
    ok_imp, note_imp = kit_importable(py)
    if not ok_imp or not py:
        return VlEvalResult(
            status="not_implemented",
            note=f"VLMEvalKit недоступен: {note_imp}. Запуск: bash scripts/setup_vlmeval_venv.sh",
            ok=False,
            error="vlmeval_missing",
        )
    if not VLMEVAL_KIT.is_dir():
        return VlEvalResult(
            status="not_implemented",
            note=f"Нет дерева {VLMEVAL_KIT}",
            ok=False,
            error="kit_missing",
        )

    datasets = [TASK_TO_DATASET[t] for t in tasks if t in TASK_TO_DATASET]
    if not datasets:
        datasets = ["DocVQA_VAL", "OCRBench"]

    work = Path(out_dir) / "kit_run"
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / "config.json"
    _write_kit_config(cfg_path, model, datasets)

    api_base = chat_completions_url(base_url)
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or "sk-local"
    env["OPENAI_API_BASE"] = api_base
    env.setdefault("CUDA_VISIBLE_DEVICES", "")

    cmd = [
        py,
        str(VLMEVAL_KIT / "run.py"),
        "--config",
        str(cfg_path),
        "--work-dir",
        str(work),
        "--mode",
        "all",
        "--api-nproc",
        "1",
        "--verbose",
    ]
    if limit is not None:
        # VLMEvalKit uses dataset-level limit via env in some forks; keep in note
        env["LLMBENCH_VL_LIMIT"] = str(limit)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(VLMEVAL_KIT),
            env=env,
            capture_output=True,
            text=True,
            timeout=43200,
        )
    except subprocess.TimeoutExpired as e:
        return VlEvalResult(
            status="error",
            note=f"VLMEvalKit timeout; limit={limit}",
            ok=False,
            error=str(e),
            raw_path=str(work),
        )

    log_path = work / "run.log"
    log_path.write_text(
        (proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8"
    )
    primary = _parse_kit_scores(work)
    if primary:
        return VlEvalResult(
            primary=primary,
            status="done",
            note=f"VLMEvalKit: {', '.join(f'{k}={v:.4f}' for k,v in primary.items())}",
            ok=True,
            raw_path=str(work),
        )
    return VlEvalResult(
        status="error",
        note=f"Прогон завершён, scores не разобраны; см. {log_path}",
        ok=False,
        error=f"no_scores exit={proc.returncode}",
        raw_path=str(work),
    )


def run_vl_eval(
    base_url: str,
    model: str,
    tasks: List[str],
    modalities: str = "text+image",
    out_dir: str = "out/vlmeval",
    limit: Optional[int] = None,
    execute: Optional[bool] = None,
    smoke_endpoint: bool = True,
) -> VlEvalResult:
    """
    execute=None → берётся LLMBENCH_VL_EXECUTE (1/true = полный kit-прогон).
    По умолчанию execute=False: только готовность адаптера, без датасетов/GPU.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not is_vision_capable(modalities):
        return VlEvalResult(
            status="skipped_text_only",
            note="не тестировалось (text-only)",
            ok=True,
        )

    norm = normalize_tasks(tasks)
    if execute is None:
        execute = os.environ.get("LLMBENCH_VL_EXECUTE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    py = vlmeval_python()
    ok_imp, note_imp = kit_importable(py)

    if smoke_endpoint:
        ok_sm, msg_sm = openai_image_url_chat(base_url, model)
        if not ok_sm:
            return VlEvalResult(
                status="error",
                note=f"image_url smoke fail: {msg_sm}",
                ok=False,
                error="smoke_fail",
            )

    if execute:
        return run_vlmevalkit(
            base_url, model, norm, out_dir=out_dir, limit=limit
        )

    datasets = [TASK_TO_DATASET[t] for t in norm if t in TASK_TO_DATASET]
    cfg_preview = Path(out_dir) / "adapter_config.preview.json"
    _write_kit_config(cfg_preview, model, datasets or ["DocVQA_VAL", "OCRBench"])

    note = (
        f"Адаптер image_url готов (GPT4V→{chat_completions_url(base_url)}, model={model}). "
        f"Задачи: {', '.join(norm)}. VLMEvalKit: {('ok' if ok_imp else note_imp)}. "
        f"Полный прогон: LLMBENCH_VL_EXECUTE=1 (отдельный .venv-vlmeval, не трогает GPU0 nightly)."
    )
    return VlEvalResult(
        status="adapter_ready",
        note=note,
        ok=True,
        error="" if ok_imp else "kit_pending_install",
        raw_path=str(cfg_preview),
    )


def merge_vision_into_quality(quality: dict, vl: VlEvalResult) -> dict:
    out = dict(quality or {})
    for k, v in (vl.primary or {}).items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out
