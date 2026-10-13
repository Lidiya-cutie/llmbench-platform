# Архитектурный обзор

## Назначение документа

Описание компонентов системы llmbench, их взаимодействия, потока данных и интеграций с внешними инструментами. Документ предназначен для разработчиков и архитекторов.

---

## Компоненты системы

### 1. CLI и скрипты (`scripts/`)

Точка входа для пользователя. Основные скрипты:

| Скрипт | Назначение |
|---|---|
| `launch_model.py` | Запуск inference-сервера (vLLM, SGLang, llama.cpp, и др.) с автоопределением движка |
| `evaluate_model.py` | Полный цикл оценки одной модели: health-check, perf, hwmon, micro-eval, lm-eval, cost, карточка |
| `compare_models.py` | Сравнение моделей с взвешенным скорингом по use-case пресетам |
| `run_with_ckpt.py` | Запуск оценки с checkpoint/resume (восстановление после сбоев) |
| `orchestrator.py` | Оркестрация множественных моделей на разных GPU |
| `auto_compare_daemon.py` | Демон автоматического сравнения после заполнения quality |
| `generate_dashboard_v2.py` | Генерация интерактивного HTML-дашборда |
| `auto_update_all_cards.py` | Автоматическое обновление карточек из новых lm-eval данных |

### 2. Ядро (`llmbench/`)

Python-пакет с модулями для измерений и анализа.

#### Цепочка оценки

```
evaluate_model.py
    |
    |-- [0/5] health.py        -- health-check эндпоинта
    |-- [1/5] perf.py          -- производительность (TTFT, tok/s, latency)
    |-- [2/5] hwmon.py         -- аппаратный мониторинг во время perf
    |-- [3/5] micro_eval.py    -- микро-тесты (math, JSON, tool use)
    |-- [4/5] lmeval.py        -- качество через lm-eval-harness
    |-- [5/5] cost.py          -- расчёт стоимости
    |
    v
modelcard.py -- агрегация всех метрик в карточку
```

#### Модуль `engine_detect.py`

Автоопределение inference-движка по конфигурации модели. Логика:

| Признак в конфиге | Движок | Порт |
|---|---|---|
| `quant_format` содержит "awq", "gptq", "BF16" | vLLM | 8077 |
| `quant_format` содержит "gguf", "ternary", "1-bit" | llama.cpp | 8080 |
| `engine: "SGLang"` | SGLang | 8078 |
| `engine: "HuggingFace TGI"` | HuggingFace TGI | 8079 |

#### Модуль `modelcard.py`

Агрегирует результаты всех измерений в единую структуру:

```json
{
  "model": { "key", "repo", "quant_format", "disk_size_gb", "context_max" },
  "perf": { "gen_tok_s_mean", "ttft_s_mean", "latency_s_p50/p90/p95/p99", ... },
  "hardware": { "peak_vram_mib", "avg_power_w", "max_temp_c", ... },
  "quality": { "mmlu_pro", "gsm8k", "humaneval", "minerva_math", ... },
  "micro": { "math_acc", "math_n" },
  "cost": { "usd_per_hour", "usd_per_1m_tokens", "usd_per_1k_requests" },
  "conclusion": "экспертное заключение"
}
```

Результат сохраняется в двух форматах:
- `out/card_{model_key}.json` — машиночитаемый
- `out/card_{model_key}.md` — человекочитаемый (Markdown)

#### Модуль `history.py`

Хранилище исторических данных оценок в SQLite (`out/history.db`). Позволяет:
- Импортировать готовые карточки в историю
- Получать тренды качества моделей за период
- Отслеживать регрессию/прогресс

#### Модуль `validate.py`

Валидация консистентности карточек:
- VRAM integrity (peak_vram > disk_size * 0.7)
- Quality не пустой
- Диапазоны качества [0, 1]
- Perf метрики (tok/s > 1, TTFT < 10s)
- Cost метрики рассчитаны

### 3. Конфигурация (`config/`)

| Файл | Назначение |
|---|---|
| `decision.yaml` | Правила принятия решений: пороги качества, скорости, стоимости для рекомендаций |
| `vision_domain.yaml` | Конфигурация vision-бенчмарков: наборы задач, параметры VLMEvalKit |

Конфигурация моделей (`models.yaml`) задаётся через конфигурационный файл с описанием каждой модели: репозиторий, формат квантизации, размер, контекст, движок, команда запуска.

### 4. Дашборд (`out/dashboard_v2.html`)

Интерактивный HTML-дашборд (single-file, без внешних зависимостей):
- Compact-таблица с ключевыми метриками всех моделей
- Expandable-секции для детализации (33 категории lm-eval, 12+ perf метрик)
- Фильтры по use-case (balanced, coding, reasoning, math, speed, cost)
- Цветовое кодирование (green/teal/orange/red/gray)

---

## Поток данных

```
1. Пользователь определяет модель в config/models.yaml
2. launch_model.py:
   a. Читает конфиг модели
   b. engine_detect.py определяет движок
   c. Генерирует launch_cmd
   d. Проверяет свободный порт (ss, docker ps)
   e. Запускает inference-сервер
   f. Сохраняет PID в logs/pids/{model_key}.pid

3. evaluate_model.py:
   a. health.py -- проверяет /v1/models
   b. perf.py -- отправляет тестовые запросы, замеряет TTFT/tok/s/latency
   c. hwmon.py -- параллельно собирает VRAM/power/temp
   d. micro_eval.py -- микро-тесты (math, JSON, tool use)
   e. lmeval.py -- запускает lm-eval-harness с выбранными задачами
   f. cost.py -- рассчитывает стоимость из power и tok/s
   g. modelcard.py -- агрегирует всё в card_{model}.json + .md

4. compare_models.py:
   a. Загружает карточки всех моделей
   b. Рассчитывает взвешенный скор по use-case
   c. Ранжирует модели
   d. Генерирует dashboard_v2.html

5. history.py (post-processing):
   a. Импортирует карточки в SQLite
   b. Рассчитывает тренды
```

---

## Интеграции

### lm-evaluation-harness (v0.4.12)

Внешний Python-пакет (`pip install lm-eval[api]`). Интеграция через `llmbench/lmeval.py`:
- Запуск задач: MMLU-Pro, GSM8K, Minerva Math, HumanEval, MBPP, GPQA, IFEval
- Парсинг результатов из JSON-вывода lm-eval
- Поддержка suites: smoke (быстрый), simplified (основной), full (полный)

**Известная проблема:** баг в `tasks/humaneval/utils.py:build_predictions()` — лишняя закрывающая скобка. Требуется локальный патч. См. [troubleshooting.md](troubleshooting.md).

### VLMEvalKit

Внешний фреймворк для оценки VLM. Интеграция через `llmbench/vlmeval.py`:
- Задачи: DocVQA, OCRBench, ChartQA
- Запуск в отдельном виртуальном окружении (setup_vlmeval_venv.sh)

### Docker

Inference-серверы запускаются преимущественно через Docker:
- vLLM: `docker run --gpus all vllm/vllm-openai ...`
- Triton: `docker run --gpus all nvcr.io/nvidia/tritonserver ...`
- Ollama: нативный сервис

### MLflow

Опциональная интеграция через `llmbench/mlflow_integration.py`:
- Логирование метрик оценки в MLflow tracking server
- Экспорт артефактов (карточек)

---

## Хранилище данных

| Путь | Назначение | Формат |
|---|---|---|
| `out/card_{model}.json` | Карточка модели (машиночитаемая) | JSON |
| `out/card_{model}.md` | Карточка модели (человекочитаемая) | Markdown |
| `out/dashboard_v2.html` | Интерактивный дашборд | HTML |
| `out/history.db` | История оценок | SQLite |
| `out/.ckpt_{model}.json` | Checkpoint оценки | JSON |
| `out/lmeval_{model}_queue/` | Сырые результаты lm-eval | JSON |
| `logs/pids/{model}.pid` | PID inference-сервера | text |
| `logs/*.log` | Логи процессов | text |

---

## Архитектурные принципы

### Backward Compatibility

Все новые модули разрабатываются как post-processing или параллельные процессы. Текущие активные оценки не прерываются.

### Изоляция ресурсов

Каждый inference-сервер работает на назначенном GPU (`CUDA_VISIBLE_DEVICES`). Оркестратор управляет распределением.

### Checkpoint/Resume

Длительные оценки (до 21 часа для simplified suite) поддерживают checkpoint/resume. Состояние сохраняется после каждого этапа.

### Reproducibility

- `requirements.lock` фиксирует точные версии зависимостей
- Конфигурации моделей в YAML — декларативные
- Результаты lm-eval сохраняются в сыром виде
