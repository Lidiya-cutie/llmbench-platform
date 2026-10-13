# Руководство разработчика

## Структура кода

### Пакет `llmbench/`

Ядро платформы. Модули:

| Модуль | Ответственность | Ключевые классы/функции |
|---|---|---|
| `hwmon.py` | GPU мониторинг | `HardwareMonitor`, `HwSummary` |
| `perf.py` | Производительность inference | `run_perf()`, `run_context_scaling()`, `PerfResult` |
| `perf_ext.py` | Расширенные perf-тесты | context scaling, batch testing |
| `micro_eval.py` | Микро-тесты | `run_micro_eval()`, `MicroResult` |
| `lmeval.py` | Интеграция lm-eval-harness | `run_lm_eval()`, `parse_results_file()`, `LmEvalResult` |
| `vlmeval.py` | Интеграция VLMEvalKit | vision-language оценка |
| `cost.py` | Расчёт стоимости | `estimate_cost()`, `CostInputs`, `CostResult` |
| `modelcard.py` | Карточка модели | `ModelCard`, `render_markdown()`, `finalize()` |
| `health.py` | Health-check эндпоинтов | `check_openai_endpoint()`, `wait_ready()`, `HealthStatus` |
| `engine_detect.py` | Определение движка | `detect_engine()`, `EngineDetection` |
| `history.py` | История оценок (SQLite) | `HistoryStore`, `EvalRun`, `import_card()` |
| `validate.py` | Валидация карточек | `CardValidator`, `validate_card()`, `validate_all()` |
| `pipeline.py` | Оркестрация пайплайна | `Pipeline` |
| `cli.py` | CLI-интерфейс | entry points в pyproject.toml |
| `mlflow_integration.py` | MLflow трекинг | опционально |
| `parquet_export.py` | Экспорт в Parquet | для аналитики |
| `ops_metrics.py` | Операционные метрики | мониторинг системы |
| `vision_integration.py` | Vision-тесты | интеграция VLM |
| `vision_agent_tests.py` | Тесты vision-агентов | agent-based оценка |

### Скрипты `scripts/`

CLI-утилиты для запуска и автоматизации.

### Тесты `tests/`

| Файл | Назначение | Количество тестов |
|---|---|---|
| `test_pure.py` | Модульные тесты без GPU | 38 |
| `test_internal.py` | Тесты internal Gate-2 отчётности | 32 |
| `test_decision.py` | Тесты принятия решений | 18 |
| `test_modelcard.py` | Тесты карточек моделей | — |
| `test_cost.py` | Тесты расчёта стоимости | — |
| `test_lmeval.py` | Тесты интеграции lm-eval | — |

---

## Добавление новой модели

### Шаг 1. Конфигурация

Добавить запись в `config/models.yaml`:

```yaml
models:
  new-model-key:
    repo: "org/model-name"
    base_model: Model-Name
    quant_format: BF16          # или AWQ, GPTQ, GGUF, ternary
    disk_size_gb: 16.0
    context_max: 32768
    modalities: text             # или text+image
    license: Apache-2.0
    engine: vLLM                 # или SGLang, llama.cpp, и др.
    served_model_name: Model-Name
    thinking_default: "auto"
    vendor_claims:
      - "Описание от вендора"
```

### Шаг 2. Проверка engine_detect

```bash
python3 -c "
from llmbench.engine_detect import detect_engine
cfg = {'repo': 'org/model-name', 'quant_format': 'BF16'}
result = detect_engine(cfg)
print(f'Engine: {result.engine}, Confidence: {result.confidence}')
"
```

Если движок определён неверно — указать `engine` явно в конфиге.

### Шаг 3. Тестовый запуск

```bash
# Запуск inference-сервера
python scripts/launch_model.py --model-key new-model-key --gpu-index 0

# Health-check
python3 -c "
from llmbench import wait_ready
status = wait_ready('http://127.0.0.1:8077/v1', 'Model-Name', timeout=180)
print(f'ok={status.ok}')
"

# Быстрая оценка (smoke suite)
python scripts/evaluate_model.py \
  --model-key new-model-key \
  --base-url http://127.0.0.1:8077/v1 \
  --suite smoke \
  --perf-n 4 \
  --perf-max-tokens 128
```

### Шаг 4. Полная оценка

```bash
python scripts/run_with_ckpt.py \
  --model-key new-model-key \
  --base-url http://127.0.0.1:8077/v1 \
  --suite simplified
```

---

## Добавление нового бенчмарка

### В lm-eval-harness

lm-eval-harness уже содержит большинство стандартных бенчмарков. Для добавления:

1. Выбрать задачи в [lm-eval task gallery](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)
2. Добавить имена задач в конфигурацию suite в `llmbench/lmeval.py`
3. При необходимости — добавить парсинг категорий в `parse_results_file()`

### Кастомный микро-тест

Добавить в `llmbench/micro_eval.py`:

```python
def run_custom_test(base_url: str, model: str) -> dict:
    """Кастомный микро-тест."""
    # Логика теста
    # HTTP-запросы к /v1/chat/completions
    # Проверка корректности ответов
    return {"custom_acc": 0.85, "custom_n": 10}
```

Интегрировать в `evaluate_model.py`:

```python
# В цикл оценки
from llmbench.micro_eval import run_custom_test
custom = run_custom_test(base_url, model_name)
card["micro"]["custom"] = custom
```

---

## Добавление Vision-бенчмарков

### Через VLMEvalKit

1. Установить VLMEvalKit в отдельное venv:

```bash
bash scripts/setup_vlmeval_venv.sh
```

2. Добавить задачу в `config/vision_domain.yaml`:

```yaml
vision_tasks:
  new_vision_task:
    dataset: "NewVisionTask"
    metric: "accuracy"
    venv: ".vlmeval-venv"
```

3. Запуск:

```bash
python scripts/evaluate_model.py \
  --model-key qwen2.5-vl-7b \
  --base-url http://127.0.0.1:8077/v1 \
  --suite simplified \
  --include-vision
```

---

## Тестирование

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# С coverage
pytest tests/ --cov=llmbench --cov-report=term-missing

# Конкретный модуль
pytest tests/test_modelcard.py -v

# С выводом печати
pytest tests/ -v -s
```

### Написание тестов

```python
# tests/test_new_feature.py
import pytest
from llmbench.new_module import new_function

class TestNewFeature:
    def test_basic(self):
        result = new_function(input_data)
        assert result is not None
        assert result.field == expected_value

    def test_edge_case(self):
        result = new_function(None)
        assert result.field == default_value
```

---

## Линтеры и форматирование

### Ruff

```bash
ruff check llmbench/ scripts/
ruff check --fix llmbench/ scripts/   # автоисправление
ruff format llmbench/ scripts/
```

### Black

```bash
black llmbench/ scripts/ --line-length 100
```

### Mypy

```bash
mypy llmbench/
```

### Обновление lockfile

```bash
pip install --upgrade pip setuptools pip-tools
pip-compile --upgrade requirements.in -o requirements.lock
# Или
pip freeze > requirements.lock
```

---

## CI/CD

### GitHub Actions

- `.github/workflows/ci.yml` — lint (ruff, black) + test matrix (Python 3.10/3.11/3.12)
- `.github/workflows/ci_v2.yml` — security scans (bandit, trufflehog) + typecheck (mypy)

### Pre-commit hooks

Хуки запускаются автоматически при `git commit`:

| Хук | Назначение |
|---|---|
| black | Форматирование кода |
| ruff | Линтинг и автоисправление |
| mypy | Проверка типов |
| pytest | Модульные тесты |
| trufflehog | Поиск секретов в коде |
| trailing-whitespace | Удаление trailing whitespace |
| end-of-file-fixer | Исправление конца файлов |
| check-yaml | Проверка YAML-синтаксиса |
| check-added-large-files | Проверка крупных файлов |

Ручной запуск:

```bash
pre-commit run --all-files
```

---

## Версионирование

Проект следует [SemVer](https://semver.org/lang/ru/):

- **MAJOR** — несовместимые изменения API
- **MINOR** — новая функциональность, обратная совместимость
- **PATCH** — исправления багов, обратная совместимость

Текущая версия указана в `pyproject.toml` → `version` и в `CHANGELOG.md`.
