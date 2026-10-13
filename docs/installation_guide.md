# Руководство по установке

## Требования к оборудованию

### Минимальные требования

| Ресурс | Минимум | Рекомендуется |
|---|---|---|
| GPU | NVIDIA, 16GB VRAM | NVIDIA RTX 4090 (24GB VRAM) |
| CUDA | 12.1 | 12.4+ |
| RAM | 32 GB | 64 GB |
| Диск | 50 GB свободного места | 200 GB (NVMe SSD) |
| CPU | 4 ядра | 8+ ядер |

### Программные требования

| Компонент | Версия |
|---|---|
| ОС | Ubuntu 22.04+ (или совместимый Linux) |
| Python | 3.10+ (рекомендуется 3.12) |
| Docker | 24.0+ |
| NVIDIA Driver | 535+ |
| NVIDIA Container Toolkit | для Docker GPU |

---

## Установка

### Шаг 1. Клонирование репозитория

```bash
git clone <repo-url> llmbench
cd llmbench
```

### Шаг 2. Виртуальное окружение Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version  # проверить: Python 3.10+
```

### Шаг 3. Установка зависимостей

#### Основные зависимости

```bash
pip install --upgrade pip setuptools
pip install -r requirements.lock
```

#### Режим разработки

```bash
pip install -e ".[dev]"
```

Устанавливает дополнительно: pytest, pytest-cov, black, ruff, mypy, pre-commit.

#### Dashboard (опционально)

```bash
pip install -e ".[dashboard]"
```

Устанавливает: streamlit, plotly, altair.

### Шаг 4. Установка inference-движков

#### vLLM (рекомендуется)

```bash
pip install vllm==0.6.4
```

Или через Docker (не требует локальной установки):

```bash
docker pull vllm/vllm-openai:latest
```

#### llama.cpp (для GGUF/ternary моделей)

```bash
# Сборка из исходников
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make GGML_CUDA=1
```

#### Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Шаг 5. Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Шаг 6. Настройка CUDA

Проверка доступности GPU:

```bash
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count}')"
```

Для Docker — установка NVIDIA Container Toolkit:

```bash
# Ubuntu
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Валидация установки

### Проверка Python-модулей

```bash
python3 -c "
from llmbench import (
    HardwareMonitor, run_perf, run_micro_eval,
    run_lm_eval, estimate_cost, ModelCard,
    check_openai_endpoint
)
print('Все модули импортированы успешно')
"
```

### Проверка тестов

```bash
pytest tests/ -v --tb=short
```

Ожидаемый результат: все тесты пройдены (test_pure, test_internal, test_decision).

### Проверка GPU

```bash
python3 -c "
from llmbench import HardwareMonitor
mon = HardwareMonitor()
stats = mon.snapshot()
print(f'GPU count: {stats.gpu_count}')
for i in range(stats.gpu_count):
    print(f'GPU{i}: {stats.vram_free_mib[i]} MiB free / {stats.vram_total_mib[i]} MiB total')
"
```

### Проверка health-check

```bash
# Если inference-сервер уже запущен
python3 -c "
from llmbench import check_openai_endpoint
status = check_openai_endpoint('http://127.0.0.1:8077/v1', timeout=10)
print(f'ok={status.ok}, latency={status.latency_ms}ms')
"
```

---

## Устранение проблем установки

### ImportError: torch

```
ModuleNotFoundError: No module named 'torch'
```

Решение: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### CUDA out of memory

```
RuntimeError: CUDA out of memory.
```

Решение: уменьшите `gpu_memory_utilization` в конфигурации модели или используйте меньший `max_model_len`.

### lm-eval: HumanEval возвращает 0%

См. [troubleshooting.md](troubleshooting.md#humaneval-возвращает-0).

---

## Переменные окружения

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | Ограничение видимых GPU | все доступные |
| `LLMBENCH_OUT_DIR` | Директория выходных данных | `out/` |
| `LLMBENCH_CONFIG_DIR` | Директория конфигурации | `config/` |
| `HF_HOME` | Кэш HuggingFace | `~/.cache/huggingface` |
| `VLLM_NO_USAGE_COLLECTION` | Отключение телеметрии vLLM | `1` |
