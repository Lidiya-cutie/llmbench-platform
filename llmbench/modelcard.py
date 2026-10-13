"""Карточка модели + генератор экспертного заключения."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

VISION_BENCHES = ("docvqa", "ocrbench", "chartqa", "mmmu")


def is_vision_capable(modalities: str | None) -> bool:
    m = (modalities or "").lower().replace(" ", "")
    return "image" in m or "vision" in m or m in ("text+image", "image+text")


@dataclass
class ModelCard:
    name: str = ""
    repo: str = ""
    base_model: Optional[str] = None
    quant_format: Optional[str] = None
    disk_size_gb: Optional[float] = None
    context_max: Optional[int] = None
    modalities: str = "text"
    license: Optional[str] = None
    engine: str = ""
    engine_version: str = ""
    serving_note: str = ""
    thinking_mode: str = ""
    quality: Dict[str, float] = field(default_factory=dict)
    micro: Dict[str, float] = field(default_factory=dict)
    agent: Dict[str, float] = field(default_factory=dict)
    perf: Dict[str, float] = field(default_factory=dict)
    context_scaling: List[dict] = field(default_factory=list)
    hardware: Dict[str, float] = field(default_factory=dict)
    cost: Dict[str, float] = field(default_factory=dict)
    reliability: Dict[str, float] = field(default_factory=dict)
    vision: Dict[str, object] = field(default_factory=dict)
    fit: Dict[str, str] = field(default_factory=dict)
    conclusion: str = ""
    gaps: List[str] = field(default_factory=list)
    vendor_claims: List[str] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


def _verdict(ok: bool, maybe: bool) -> str:
    if ok:
        return "подходит"
    if maybe:
        return "с оговорками"
    return "не рекомендуется"


def assess_fit(card: ModelCard) -> Dict[str, str]:
    """Вердикты строго из метрик."""
    q = card.quality or {}
    micro = card.micro or {}
    agent = card.agent or {}
    perf = card.perf or {}

    def _q(*keys: str) -> Optional[float]:
        for k in keys:
            v = q.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def _avg(vals):
        xs = [float(v) for v in vals if isinstance(v, (int, float))]
        return sum(xs) / len(xs) if xs else None

    ttft = perf.get("ttft_s_mean")
    gen = perf.get("gen_tok_s_mean")
    json_ok = micro.get("json_valid")
    tool_ok = micro.get("tool_call_ok")
    multi_tool = agent.get("multi_tool_score")
    multi_tool_n = agent.get("multi_tool_n")
    ifeval = _q("ifeval")
    mmlu = _q("mmlu_pro")
    gpqa = _q("gpqa")
    he = _q("humaneval")
    mbpp = _q("mbpp")
    reason = _avg([x for x in (mmlu, gpqa) if x is not None])
    code_best = max([x for x in (he, mbpp) if x is not None], default=None)

    fit: Dict[str, str] = {}
    chat_ok = bool(
        (ifeval is not None and ifeval >= 0.55)
        or (json_ok is not None and json_ok >= 0.8)
        or (gen is not None and gen >= 25 and (ttft is None or ttft <= 2.0))
    )
    chat_maybe = bool(gen is not None and gen >= 15)
    fit["chat"] = _verdict(chat_ok, chat_maybe)

    code_ok = bool(code_best is not None and code_best >= 0.45)
    code_maybe = bool(code_best is not None and code_best >= 0.25)
    fit["coding"] = _verdict(code_ok, code_maybe)

    reason_ok = bool(reason is not None and reason >= 0.35)
    reason_maybe = bool(reason is not None and reason >= 0.2)
    fit["reasoning"] = _verdict(reason_ok, reason_maybe)

    tool_ok_b = bool(tool_ok is not None and tool_ok >= 0.5) or bool(
        agent.get("bfcl_overall") and float(agent["bfcl_overall"]) >= 0.4
    )
    tool_maybe = bool(tool_ok is not None) or bool(multi_tool_n)
    fit["tools"] = _verdict(tool_ok_b, tool_maybe)

    if is_vision_capable(card.modalities):
        vis = card.vision or {}
        scores = [
            float(vis[k])
            for k in VISION_BENCHES
            if isinstance(vis.get(k), (int, float))
        ]
        vavg = sum(scores) / len(scores) if scores else None
        vok = bool(vavg is not None and vavg >= 0.4)
        vmaybe = bool(vavg is not None and vavg >= 0.2) or vis.get("status") in (
            "adapter_ready",
            "pending",
            "done",
        )
        fit["vision_ocr"] = _verdict(vok, vmaybe)
    else:
        fit["vision_ocr"] = "не тестировалось"

    return fit


def build_conclusion(card: ModelCard) -> str:
    q = card.quality or {}
    perf = card.perf or {}
    hw = card.hardware or {}
    micro = card.micro or {}
    parts: List[str] = []

    mem = hw.get("peak_vram_mb")
    disk = card.disk_size_gb
    ctx = card.context_max
    quant = card.quant_format or "формат не указан"
    if mem:
        gb = float(mem) / 1024.0
        parts.append(
            f"{card.name} ({quant}"
            + (f", {disk} ГБ на диске)" if disk else ")")
            + f" на RTX 4090 занимает порядка {gb:.1f} ГБ VRAM в пике"
            + (f" и оставляет запас под контекст (до {ctx} токенов)." if ctx else ".")
        )
    else:
        parts.append(f"{card.name} оценивалась на RTX 4090.")

    gen = perf.get("gen_tok_s_mean")
    ttft = perf.get("ttft_s_mean")
    if gen:
        speed_word = "достаточно" if gen >= 30 else "может не хватать"
        parts.append(
            f"Скорость генерации — {gen:.1f} ток/с"
            + (f", TTFT ~{ttft:.2f} c" if ttft is not None else "")
            + f"), чего {speed_word} для интерактивных чат-сценариев."
        )

    he = q.get("humaneval")
    mmlu = q.get("mmlu_pro")
    gsm = q.get("gsm8k")
    quality_bits = []
    if isinstance(mmlu, (int, float)):
        quality_bits.append(
            f"reasoning {'уверенный' if mmlu >= 0.4 else 'ограниченный'} (MMLU-Pro {mmlu:.3f})"
        )
    if isinstance(he, (int, float)):
        quality_bits.append(f"coding {he:.3f}")
    if isinstance(gsm, (int, float)):
        quality_bits.append(f"арифметика {gsm:.3f}")
    if quality_bits:
        parts.append("По качеству: " + "; ".join(quality_bits) + ".")
    else:
        parts.append(
            "Стандартизированные бенчмарки качества (MMLU-Pro/GPQA/HumanEval) "
            "в этом прогоне не запускались — оценка опирается на micro-eval."
        )

    if is_vision_capable(card.modalities):
        vis = card.vision or {}
        st = vis.get("status", "pending")
        bits = []
        for k in VISION_BENCHES:
            v = vis.get(k)
            if isinstance(v, (int, float)):
                bits.append(f"{k}={float(v):.3f}")
        if bits:
            parts.append("Vision/OCR: " + ", ".join(bits) + f" (status={st}).")
        else:
            parts.append(f"Vision/OCR: status={st}; DocVQA/OCRBench ещё не закрыты.")

    fit = card.fit or {}
    good = [k for k, v in fit.items() if v == "подходит"]
    soft = [k for k, v in fit.items() if v == "с оговорками"]
    weak = [k for k, v in fit.items() if v == "не рекомендуется"]
    if good:
        parts.append("Рекомендуется для: " + ", ".join(good) + ".")
    if soft:
        parts.append("Подходит с оговорками для: " + ", ".join(soft) + ".")
    if weak:
        parts.append("Не рекомендуется для: " + ", ".join(weak) + ".")

    if good and not weak:
        parts.append("ИТОГО: РЕКОМЕНДОВАНО только для " + ", ".join(good) + ".")
    elif soft and not good:
        parts.append("ИТОГО: узкая рекомендация — " + ", ".join(soft) + ".")
    elif weak and not good:
        parts.append(
            "ИТОГО: НЕ РЕКОМЕНДУЕТСЯ как универсальная production-модель; "
            "допустима с оговорками для: " + (", ".join(soft) if soft else "узких тестов") + "."
        )
    return " ".join(parts)


_TEMPLATE = """# Карточка модели: {{ c.name }}

_Сгенерировано: {{ c.generated_at }} · Стенд: RTX 4090_

## 1. Идентификация
| Поле | Значение |
|---|---|
| Репозиторий | `{{ c.repo }}` |
| Базовая модель | {{ c.base_model or '—' }} |
| Квантизация | {{ c.quant_format or '—' }} |
| Модальности | {{ c.modalities }} |
| Движок | {{ c.engine }} {{ c.engine_version }} |

## 2. Качество (core-7)
{% for k, v in c.quality.items() %}
- {{ k }}: {{ v }}
{% endfor %}

## 3. Vision / OCR
{% if c.vision %}
{% for k, v in c.vision.items() %}
- {{ k }}: {{ v }}
{% endfor %}
{% else %}
- не заполнено
{% endif %}

## 4. Заключение
{{ c.conclusion }}
"""


def render_markdown(card: ModelCard) -> str:
    import jinja2

    env = jinja2.Environment(autoescape=False)
    return env.from_string(_TEMPLATE).render(c=card)


def finalize(card: ModelCard) -> ModelCard:
    """Досчитать fit + conclusion по уже собранным метрикам."""
    card.fit = assess_fit(card)
    card.conclusion = build_conclusion(card)
    return card
