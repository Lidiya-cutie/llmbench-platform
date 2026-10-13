"""
Автономный micro-eval качества (без lm-eval-harness).

Даёт быстрый сигнал сразу после запуска сервера: арифметическое рассуждение,
валидность структурированного вывода (JSON) и корректность tool-calling.
Скоринг детерминированный (никакого LLM-судьи), поэтому воспроизводим.

Для полноценной оценки (MMLU-Pro, GPQA, HumanEval, RULER, BFCL...) используйте
модуль lmeval.py — этот файл лишь sanity-слой.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import requests

# ----------------------------- пул задач ------------------------------------

MATH_ITEMS = [
    ("У Ани было 3 коробки по 12 карандашей и ещё 7 карандашей. Сколько всего карандашей? "
     "В конце напиши: Ответ: <число>.", 43),
    ("Поезд едет 4 часа со скоростью 65 км/ч. Сколько км он проедет? Ответ: <число>.", 260),
    ("В классе 28 учеников, 3/4 сдали экзамен. Сколько сдали? Ответ: <число>.", 21),
    ("Магазин снизил цену 200 руб на 15%. Новая цена? Ответ: <число>.", 170),
    ("17*23 = ? Ответ: <число>.", 391),
]

JSON_ITEM = (
    'Верни ТОЛЬКО валидный JSON без пояснений и markdown, строго по схеме: '
    '{"city": string, "country": string, "population_millions": number}. '
    'Данные: Токио, Япония, ~37 млн.'
)

TOOL_ITEM = "Какая погода сейчас в Лиссабоне? Используй доступную функцию."
TOOLS_SPEC = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Текущая погода в городе",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


@dataclass
class MicroResult:
    math_acc: float
    math_n: int
    json_valid: float          # 0/1: распарсился и поля на месте
    tool_call_ok: float        # 0/1: вызвал get_weather с city~Lisbon
    multi_tool_score: float    # 0-1: многошаговые сценарии
    multi_tool_n: int         # количество протестированных сценариев
    detail: dict

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------- чистые функции скоринга (тестируемы) ------------------

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_final_number(text: str) -> Optional[float]:
    """Достаём числовой ответ: приоритет \\boxed{}, затем 'Ответ:/answer', затем последнее число."""
    if not text:
        return None
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        n = _NUM.search(m.group(1))
        if n:
            return float(n.group().replace(",", "."))
    m = re.search(r"(?:ответ|answer)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)", text, re.I)
    if m:
        return float(m.group(1).replace(",", "."))
    nums = _NUM.findall(text)
    return float(nums[-1].replace(",", ".")) if nums else None


def score_json(text: str) -> bool:
    """Модель должна вернуть валидный JSON с нужными полями (терпим к обёрткам)."""
    if not text:
        return False
    s = text.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.I | re.M).strip()
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group())
    except Exception:
        return False
    return (isinstance(obj.get("city"), str)
            and isinstance(obj.get("country"), str)
            and isinstance(obj.get("population_millions"), (int, float)))


def score_tool_call(message: dict) -> bool:
    """Проверяем, что модель вызвала get_weather с городом Лиссабон."""
    calls = (message or {}).get("tool_calls") or []
    for c in calls:
        fn = c.get("function", {})
        if fn.get("name") == "get_weather":
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            city = str(args.get("city", "")).lower()
            if "lisb" in city or "лисс" in city:
                return True
    return False


def score_multi_tool_scenario(message: dict, scenario: dict) -> bool:
    """Проверяет успешность многошагового сценария."""
    try:
        calls = (message or {}).get("tool_calls") or []
        check_fn = scenario.get("check_fn")
        if check_fn:
            return check_fn(calls)
        return False
    except Exception as e:
        print(f"Error scoring scenario {scenario.get('name')}: {e}")
        return False


# ---------------------------- вызов эндпоинта -------------------------------

def _chat(base_url: str, model: str, messages: List[dict], *, tools=None,
          max_tokens: int = 512, timeout: float = 300.0, extra_body: dict = None) -> dict:
    payload = {"model": model, "messages": messages, "temperature": 0.0,
               "max_tokens": max_tokens}
    if tools:
        payload["tools"] = tools
    payload.update(extra_body or {})
    r = requests.post(base_url.rstrip("/") + "/chat/completions",
                      json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def run_micro_eval(base_url: str, model: str, *, extra_body: dict = None,
                   run_multi_tool: bool = True) -> MicroResult:
    # математика
    correct, detail_math = 0, []
    for q, gold in MATH_ITEMS:
        try:
            msg = _chat(base_url, model, [{"role": "user", "content": q}], extra_body=extra_body)
            got = extract_final_number(msg.get("content", ""))
            ok = got is not None and abs(got - gold) < 1e-6
        except Exception:
            got, ok = None, False
        correct += int(ok)
        detail_math.append({"gold": gold, "got": got, "ok": ok})

    # JSON
    try:
        jmsg = _chat(base_url, model, [{"role": "user", "content": JSON_ITEM}], extra_body=extra_body)
        json_ok = score_json(jmsg.get("content", ""))
    except Exception:
        json_ok = False

    # tool calling (одиночный вызов)
    try:
        tmsg = _chat(base_url, model, [{"role": "user", "content": TOOL_ITEM}],
                     tools=TOOLS_SPEC, extra_body=extra_body)
        tool_ok = score_tool_call(tmsg)
    except Exception:
        tool_ok = False

    # многошаговые сценарии
    multi_tool_score = 0.0
    multi_tool_n = 0
    detail_multi = []

    if run_multi_tool:
        try:
            from .multi_tool_scenarios import MULTI_TOOL_SCENARIUS

            for scenario in MULTI_TOOL_SCENARIUS:
                try:
                    mtmsg = _chat(base_url, model, scenario["messages"],
                                   tools=scenario["tools"], extra_body=extra_body)
                    scenario_ok = score_multi_tool_scenario(mtmsg, scenario)
                    multi_tool_n += 1
                    if scenario_ok:
                        multi_tool_score += 1
                    detail_multi.append({
                        "scenario": scenario["name"],
                        "ok": scenario_ok,
                        "expected_calls": scenario.get("expected_calls", 0),
                        "actual_calls": len(mtmsg.get("tool_calls", []))
                    })
                except Exception as e:
                    multi_tool_n += 1
                    detail_multi.append({
                        "scenario": scenario.get("name", "unknown"),
                        "ok": False,
                        "error": str(e)
                    })

            if multi_tool_n > 0:
                multi_tool_score = round(multi_tool_score / multi_tool_n, 3)
        except ImportError:
            # multi_tool_scenarios.py не найден - пропускаем
            multi_tool_n = 0

    return MicroResult(
        math_acc=round(correct / len(MATH_ITEMS), 3),
        math_n=len(MATH_ITEMS),
        json_valid=float(json_ok),
        tool_call_ok=float(tool_ok),
        multi_tool_score=multi_tool_score,
        multi_tool_n=multi_tool_n,
        detail={"math": detail_math, "multi_tool": detail_multi},
    )
