# Руководство по развёртыванию

## Сценарии развёртывания

### 1. Развёртывание на одном GPU (RTX 4090)

Базовый сценарий: одна модель на одном GPU.

```bash
# 1. Запуск inference-сервера
python scripts/launch_model.py \
  --model-key qwen3-8b-bf16 \
  --gpu-index 0 \
  --port 8077

# 2. Проверка готовности
python3 -c "
from llmbench import wait_ready
status = wait_ready('http://127.0.0.1:8077/v1', 'Qwen3-8B', timeout=180)
print(f'ok={status.ok}, latency={status.latency_ms}ms')
"

# 3. Запуск оценки
python scripts/evaluate_model.py \
  --model-key qwen3-8b-bf16 \
  --base-url http://127.0.0.1:8077/v1 \
  --suite simplified \
  --perf-n 12 \
  --perf-max-tokens 256

# 4. Остановка
python scripts/launch_model.py --stop qwen3-8b-bf16
```

### 2. Развёртывание на двух GPU

Параллельная оценка двух моделей.

```bash
# GPU0: первая модель
python scripts/launch_model.py --model-key qwen3-8b-bf16 --gpu-index 0 --port 8077 &

# GPU1: вторая модель (параллельно)
python scripts/launch_model.py --model-key local-27b-1bit --gpu-index 1 --port 8080 &

# Оценка через оркестратор
python scripts/orchestrator.py \
  --model local-27b-1bit \
  --gpu 1 \
  --suite simplified \
  --max-parallel 2
```

### 3. Удалённый inference-сервер

Модель развёрнута на отдельном сервере, оценка — локально.

```bash
# На inference-сервере: запуск vLLM
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000

# Локально: оценка
python scripts/evaluate_model.py \
  --model-key qwen3-8b-bf16 \
  --base-url http://<SERVER_IP>:8000/v1 \
  --suite simplified \
  --skip-health-check
```

---

## Docker-контейнеры

### vLLM

```bash
docker run -d \
  --name vllm-qwen3 \
  --gpus '"device=0"' \
  -p 8077:8000 \
  -v /opt/llm-data/models:/models \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --served-model-name Qwen3-8B \
  --trust-remote-code
```

### Triton Inference Server

```bash
docker run -d \
  --name tritonserver \
  --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /opt/llm-data/triton/model_repo:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### llama.cpp

```bash
docker run -d \
  --name llama-server \
  --gpus '"device=1"' \
  -p 8080:8080 \
  -v /opt/llm-data/models:/models \
  ghcr.io/ggerganov/llama.cpp:server \
  -m /models/local-27b-27b.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 4096
```

---

## Настройка сети и портов

### Стандартные порты

| Сервис | Порт | Назначение |
|---|---|---|
| vLLM | 8077 | OpenAI-совместимый API |
| SGLang | 8078 | OpenAI-совместимый API |
| HuggingFace TGI | 8079 | OpenAI-совместимый API |
| llama.cpp | 8080 | OpenAI-совместимый API |
| TensorRT-LLM | 8082 | OpenAI-совместимый API |
| Ollama | 11434 | Ollama API |
| Triton | 8000/8001/8002 | HTTP/gRPC/metrics |
| Дашборд | — | статичный HTML файл |

### Проверка занятости портов

```bash
# Системные порты
ss -tulpn | grep -E '8077|8080|8000'

# Docker контейнеры
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

### Конфликты портов

При запуске через `launch_model.py` проверяется занятость порта. Если порт занят:

```bash
$ python scripts/launch_model.py --model-key qwen3-8b --port 8077
# Порт 8077 занят. Используйте --port или остановите процесс.
```

---

## Конфигурация моделей

Модели описываются в `config/models.yaml`:

```yaml
models:
  qwen3-8b-bf16:
    repo: "Qwen/Qwen3-8B"
    base_model: Qwen3-8B
    quant_format: BF16
    disk_size_gb: 16.0
    context_max: 32768
    modalities: text
    license: Apache-2.0
    engine: vLLM
    served_model_name: Qwen3-8B
    launch_cmd: "vllm serve /path/to/model --host 0.0.0.0 --port 8077 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 --served-model-name Qwen3-8B"
    thinking_default: "auto"
```

### Параметры

| Поле | Тип | Назначение |
|---|---|---|
| `repo` | string | HuggingFace репозиторий |
| `quant_format` | string | Формат квантизации (BF16, AWQ, GPTQ, GGUF, ternary) |
| `disk_size_gb` | float | Размер весов на диске (GB) |
| `context_max` | int | Максимальная длина контекста (токены) |
| `modalities` | string | Модальности: text, text+image |
| `engine` | string | Inference-движок (vLLM, SGLang, llama.cpp, ...) |
| `launch_cmd` | string | Команда запуска (генерируется автоматически через engine_detect) |
| `served_model_name` | string | Имя модели в API |

---

## Suites оценки

| Suite | Задачи | Время | Назначение |
|---|---|---|---|
| `smoke` | gsm8k (ограничено) | ~30 мин | Быстрая проверка работоспособности |
| `simplified` | gsm8k, minerva_math, mmlu_pro, humaneval | 12-24 часа | Основная оценка |
| `full` | все доступные задачи | 24-48 часов | Полная оценка |

---

## Запуск бенчмарков

### Одиночная модель

```bash
python scripts/evaluate_model.py \
  --model-key qwen3-8b-bf16 \
  --base-url http://127.0.0.1:8077/v1 \
  --suite simplified \
  --perf-n 12 \
  --perf-max-tokens 256
```

### С checkpoint/resume

```bash
# Первый запуск
python scripts/run_with_ckpt.py \
  --model-key qwen3-8b-bf16 \
  --base-url http://127.0.0.1:8077/v1 \
  --suite simplified

# Возобновление после сбоя
python scripts/run_with_ckpt.py --resume \
  --model-key qwen3-8b-bf16
```

### Сравнение моделей

```bash
python scripts/compare_models.py \
  --baseline qwen3-8b-bf16 \
  --models glm-4-9b-chat,qwen2.5-vl-7b \
  --use-case balanced
```

### Ночные прогоны (cron)

```bash
# crontab -e
0 2 * * * cd /path/to/llmbench && bash scripts/run_nightly_compare_fixed.sh
*/30 * * * * cd /path/to/llmbench && bash scripts/run_full_auto_update.sh
```

---

## Мониторинг

### System Monitor

```bash
# HTML-отчёт состояния системы
python scripts/system_monitor_v2.py --html
# Результат: out/system_status_v2.html (обновляется каждые 5 минут)
```

### Логи

```bash
# Логи оценки
tail -f logs/eval_*.log

# Логи inference-сервера
tail -f logs/*_server.log

# Логи автоматического обновления
tail -f logs/auto_update.log
```

### GPU мониторинг

```bash
# Непрерывный мониторинг
watch -n 1 nvidia-smi

# Через llmbench
python3 -c "
from llmbench import HardwareMonitor
mon = HardwareMonitor()
while True:
    s = mon.snapshot()
    for i in range(s.gpu_count):
        print(f'GPU{i}: {s.vram_used_mib[i]} MiB, {s.util_pct[i]}% util')
"
```
