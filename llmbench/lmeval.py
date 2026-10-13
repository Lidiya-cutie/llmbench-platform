"""
Тонкая обёртка над lm-evaluation-harness (EleutherAI).

Мы НЕ переписываем MMLU/GSM8K/GPQA сами — это главный источник расхождений в
цифрах (few-shot, форматирование, извлечение ответа). Вместо этого гоняем
официальный harness в режиме `local-chat-completions` против нашего эндпоинта
и парсим results.json. Так цифры сопоставимы с публичными таблицами.

Установка на боксе с 4090:
    pip install "lm-eval[api]"
Пример ручного запуска (то, что делает этот модуль):
    lm_eval --model local-chat-completions \\
      --model_args base_url=http://localhost:8080/v1/chat/completions,model=local-27b,num_concurrent=1,tokenized_requests=False \\
      --tasks gsm8k,mmlu_pro \\
      --apply_chat_template --output_path out/lmeval --log_samples

Для VLM (DocVQA/OCRBench/MMMU) используйте VLMEvalKit, для tool-calling — BFCL;
оба тоже общаются по OpenAI-протоколу и подключаются тем же base_url.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LmEvalResult:
    tasks: Dict[str, dict] = field(default_factory=dict)   # task -> {metric: value}
    primary: Dict[str, float] = field(default_factory=dict)  # task -> главная метрика
    raw_path: str = ""
    ok: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {"primary": self.primary, "tasks": self.tasks,
                "ok": self.ok, "error": self.error, "raw_path": self.raw_path}


_PREFERRED_METRICS = (
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "math_verify,none",
    "exact_match,custom-extract",
    "pass@1,create_test",
    "pass@1,execute",
    "acc_norm,none", "acc,none", "exact_match", "acc_norm", "acc",
)


def _pick_primary(metrics: dict) -> float:
    for k in _PREFERRED_METRICS:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return float(metrics[k])
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not k.endswith("_stderr"):
            return float(v)
    return 0.0


def run_lm_eval(base_url: str, model: str, tasks: List[str], *,
                out_dir: str = "out/lmeval", num_fewshot: int = None,
                limit: int = None, apply_chat_template: bool = True,
                num_concurrent: int = 1, extra_cli: List[str] = None,
                timeout: float = 60 * 60 * 6) -> LmEvalResult:
    os.makedirs(out_dir, exist_ok=True)
    model_args = (f"base_url={base_url.rstrip('/')}/chat/completions,"
                  f"model={model},num_concurrent={num_concurrent},"
                  f"tokenized_requests=False")
    cmd = ["lm_eval", "--model", "local-chat-completions",
           "--model_args", model_args,
           "--tasks", ",".join(tasks),
           "--output_path", out_dir]
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if extra_cli:
        cmd += extra_cli

    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except Exception as e:
        return LmEvalResult(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")

    files = sorted(glob.glob(os.path.join(out_dir, "**", "results*.json"), recursive=True),
                   key=os.path.getmtime)
    if not files:
        return LmEvalResult(ok=False, error="results.json не найден")
    return parse_results_file(files[-1])


def parse_results_file(path: str) -> LmEvalResult:
    """Отделено от запуска, чтобы тестировать парсинг на фикстурах."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return LmEvalResult(ok=False, error=f"parse: {e}")
    res = data.get("results", {})
    tasks, primary = {}, {}
    for task, metrics in res.items():
        clean = {k: v for k, v in metrics.items() if k != "alias"}
        tasks[task] = clean
        primary[task] = round(_pick_primary(clean), 4)
    return LmEvalResult(tasks=tasks, primary=primary, raw_path=path,
                        ok=bool(primary), error="" if primary else "пустые результаты")
