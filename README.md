# llmbench

**Local LLM benchmarking platform: quality + performance + hardware + cost on a single GPU.**

English TL;DR: llmbench benchmarks locally deployed LLMs (RTX 4090 24GB class) across
quality benchmarks (MMLU-Pro, GSM8K, HumanEval, MBPP, GPQA via lm-evaluation-harness;
BFCL for tool use; VLMEvalKit for vision-language), inference performance (TTFT,
tokens/sec, latency percentiles), hardware telemetry (VRAM, power, temperature) and
inference cost. Results are aggregated into model cards, validated, stored with history
trends (SQLite) and tracked in MLflow; an interactive dashboard supports model selection
decisions. Includes decision rules config, health-checks, CI and pytest suite.

---

Платформа сравнительного анализа локальных больших языковых моделей (LLM) на GPU.

Версия: **0.2.0** (Alpha)

## Назначение

llmbench предназначен для автоматизированного бенчмаркинга LLM, развёрнутых на локальных GPU (первично — NVIDIA RTX 4090 24GB). Платформа измеряет качество моделей на стандартных бенчмарках (MMLU-Pro, GSM8K, HumanEval, MBPP, GPQA и др.), производительность inference (TTFT, токены/сек, latency), аппаратные метрики (VRAM, мощность, температура) и стоимость inference. Результаты агрегируются в карточки моделей и визуализируются в интерактивном дашборде для принятия решений о выборе модели.

## Спецификация комплекса

| Компонент           | Назначение                                                      |
|---------------------|-----------------------------------------------------------------|
| `llmbench/`         | Python-пакет ядро: мониторинг, perf-тесты, eval, карточки       |
| `scripts/`          | CLI-скрипты: запуск оценки, сравнение, оркестрация, дашборд    |
| `config/`           | YAML-конфигурации: модели, пороги, правила принятия решений     |
| `tests/`            | Модульные тесты (pytest)                                        |
| `out/`              | Выходные данные: карточки, отчёты, дашборды                     |
| `docs/`             | Документация: установка, развёртывание, разработка, архитектура|
| `.github/workflows/`| CI/CD пайплайны                                                 |

### Состав пакета `llmbench/`

| Модуль                | Назначение                                                    |
|-----------------------|---------------------------------------------------------------|
| `hwmon.py`            | Аппаратный мониторинг GPU (VRAM, power, util, temp)          |
| `perf.py`             | Измерение производительности (TTFT, tok/s, latency, percentiles)|
| `perf_ext.py`         | Расширенные perf-тесты (context scaling, batch)              |
| `micro_eval.py`       | Микро-тесты (арифметика, JSON, tool use)                     |
| `lmeval.py`           | Интеграция с lm-evaluation-harness                            |
| `vlmeval.py`          | Интеграция с VLMEvalKit (vision-language модели)             |
| `cost.py`             | Расчёт стоимости inference                                    |
| `modelcard.py`        | Структура карточки модели и рендеринг Markdown                |
| `health.py`           | Health-check OpenAI-совместимых эндпоинтов                    |
| `engine_detect.py`    | Автоопределение inference-движка по конфигурации модели       |
| `history.py`          | Хранение истории оценок в SQLite, тренды                      |
| `validate.py`         | Валидация консистентности карточек                            |
| `pipeline.py`         | Оркестрация пайплайна оценки                                  |
| `cli.py`              | CLI-интерфейс                                                 |
| `mlflow_integration.py`| Интеграция с MLflow                                          |
| `parquet_export.py`   | Экспорт результатов в Parquet                                 |
| `ops_metrics.py`      | Операционные метрики                                          |
| `vision_integration.py`| Интеграция vision-тестов                                     |
| `vision_agent_tests.py`| Тесты vision-агентов                                         |

## Ключевые возможности

- **Оценка качества**: MMLU-Pro (24 категории), GSM8K, Minerva Math (9 категорий), HumanEval, MBPP, GPQA, IFEval
- **Vision-оценка**: DocVQA, OCRBench, ChartQA через VLMEvalKit
- **Производительность**: TTFT, токены/сек (median/mean), latency p50/p90/p95/p99, prefill speed, context scaling
- **Аппаратный мониторинг**: VRAM (peak/avg), power (avg/max), GPU utilisation, температура
- **Стоимость**: USD/час, USD/1M токенов, USD/1K запросов, Wh/1K токенов
- **Сравнение моделей**: взвешенный скор по use-case пресетам (balanced, coding, reasoning, math, speed, cost)
- **История и тренды**: SQLite хранилище, регрессия/прогресс моделей
- **Оркестрация GPU**: параллельная оценка на нескольких GPU, автоопределение свободных
- **Дашборд**: интерактивный HTML с фильтрами, сортировкой, expandable секциями
- **Автоматизация**: cron-задачи, checkpoint/resume, auto-compare

## Архитектура

Подробное описание: [docs/architecture.md](docs/architecture.md).

```
Запрос на оценку
       |
       v
[launch_model.py] --запуск--> vLLM / SGLang / llama.cpp / Triton
       |
       v
[health.py] --проверка--> HTTP /v1/models endpoint
       |
       v
[evaluate_model.py] --оркестрация-->
       |
       +-- [perf.py] ---------> Производительность (TTFT, tok/s, latency)
       +-- [hwmon.py] --------> Аппаратный мониторинг (VRAM, power, temp)
       +-- [micro_eval.py] ---> Микро-тесты (math, JSON, tool use)
       +-- [lmeval.py] -------> lm-eval-harness (MMLU-Pro, GSM8K, HumanEval...)
       +-- [vlmeval.py] ------> VLMEvalKit (DocVQA, OCRBench, ChartQA...)
       +-- [cost.py] ---------> Расчёт стоимости
       |
       v
[modelcard.py] --агрегация--> card_{model}.json + card_{model}.md
       |
       v
[compare_models.py] --сравнение--> dashboard_v2.html
       |
       v
[history.py] --сохранение--> SQLite (тренды, регрессия)
```

## Быстрый старт

### Требования

- Python 3.10+
- NVIDIA GPU (минимум 16GB VRAM, рекомендуется 24GB)
- CUDA 12.1+
- Docker (для контейнерных inference-серверов)

### Установка

```bash
# Клонирование
git clone <repo-url> llmbench
cd llmbench

# Виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# Зависимости
pip install -r requirements.lock

# Режим разработки
pip install -e ".[dev]"
```

Подробно: [docs/installation_guide.md](docs/installation_guide.md).

### Запуск оценки

```bash
# 1. Запуск inference-сервера
python scripts/launch_model.py --model-key qwen3-8b-bf16 --gpu-index 0

# 2. Оценка модели
python scripts/evaluate_model.py \
  --model-key qwen3-8b-bf16 \
  --base-url http://127.0.0.1:8077/v1 \
  --suite simplified

# 3. Сравнение моделей
python scripts/compare_models.py \
  --baseline qwen3-8b-bf16 \
  --models glm-4-9b-chat,qwen2.5-vl-7b

# 4. Дашборд
python scripts/generate_dashboard_v2.py
# Открыть: out/dashboard_v2.html
```

### Тестирование

```bash
pytest tests/ -v
pytest tests/ --cov=llmbench --cov-report=term-missing
```

## Структура проекта

```
llmbench/
|-- llmbench/              # Python-пакет
|-- scripts/               # CLI-скрипты и автоматизация
|-- config/                # YAML-конфигурации
|   |-- decision.yaml      # Правила принятия решений
|   `-- vision_domain.yaml # Конфигурация vision-бенчмарков
|-- tests/                 # Модульные тесты
|-- docs/                  # Документация
|   |-- installation_guide.md
|   |-- deployment_guide.md
|   |-- developer_guide.md
|   |-- architecture.md
|   |-- troubleshooting.md
|   `-- archive/           # Устаревшие .md файлы (сохранены для истории)
|-- out/                   # Выходные данные (карточки, дашборды)
|-- .github/workflows/     # CI/CD
|-- pyproject.toml         # Метаданные пакета
|-- requirements.lock      # Зафиксированные зависимости
|-- .pre-commit-config.yaml
|-- glossary.md            # Глоссарий
|-- CHANGELOG.md           # История изменений
`-- README.md              # Этот файл
```

## Документация

| Документ | Назначение |
|---|---|
| [docs/installation_guide.md](docs/installation_guide.md) | Руководство по установке |
| [docs/deployment_guide.md](docs/deployment_guide.md) | Руководство по развёртыванию |
| [docs/developer_guide.md](docs/developer_guide.md) | Руководство разработчика |
| [docs/architecture.md](docs/architecture.md) | Архитектурный обзор |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Устранение неполадок |
| [glossary.md](glossary.md) | Глоссарий терминов |
| [CHANGELOG.md](CHANGELOG.md) | История изменений |

## Поддерживаемые inference-движки

| Движок | Форматы | Порт по умолчанию |
|---|---|---|
| vLLM | BF16, FP16, AWQ, GPTQ | 8077 |
| SGLang | BF16, FP16 | 8078 |
| llama.cpp | GGUF, ternary, 1-bit | 8080 |
| HuggingFace TGI | BF16, FP16 | 8079 |
| Ollama | GGUF | 11434 |
| TensorRT-LLM | TRT-оптимизированные | 8082 |

## CI/CD

- `.github/workflows/ci.yml` — базовый CI (lint + test matrix)
- `.github/workflows/ci_v2.yml` — расширенный CI (security + typecheck)
- `.pre-commit-config.yaml` — pre-commit hooks (black, ruff, mypy, pytest, trufflehog)

## Лицензия

Apache-2.0
