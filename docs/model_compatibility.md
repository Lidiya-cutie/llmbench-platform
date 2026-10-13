# Совместимость моделей и версий vLLM

Документ описывает требования к образам vLLM для разных классов моделей, известные несовместимости и рекомендации по запуску на NVIDIA RTX 4090 (24 GB VRAM).

## Текущая инфраструктура

| Компонент | Значение |
|---|---|
| GPU | 2x NVIDIA GeForce RTX 4090 (24564 MiB каждая) |
| Образ (production) | `mlops-triton-vllm:24.12-qwen3` (vLLM 0.8.5.post1) |
| Образ (test) | `vllm/vllm-openai:v0.10.2` |
| GPU0, порт 8070 | text LLM (GLM-4-9B-Chat) |
| GPU1, порт 8077 | test VLM |
| Дашборд | порт 8899 |

## Поддержка моделей по версиям vLLM

### vLLM 0.8.5.post1 (образ `mlops-triton-vllm:24.12`)

**Поддерживаются:**
- Text LLM: Qwen2.5/Qwen3, GLM-4, DeepSeek-V2/V3, Llama 3.x, Gemma 2
- VLM: Qwen2.5-VL, InternVL3 (не 3.5!), DeepSeekVL-V2, LLaVA

**Не поддерживаются:**
- InternVL3.5 (любой формат) — `InternVLConfig.vocab_size` отсутствует в конфиге
- Janus-Pro — тип модели `multi_modality` не распознан
- Molmo2 — нет в реестре

### vLLM 0.10.2 (образ `vllm/vllm-openai:v0.10.2`)

**Поддерживаются:**
- Все модели из 0.8.5
- InternVL3.5 (`InternVLForConditionalGeneration`) — с оговорками (см. ниже)
- Molmo (`MolmoForCausalLM`)

**Известные проблемы:**
- InternVL3.5-HF: `pixel_values dim[2] expected 448, got 384` — баг обработки изображений
- vLLM 0.10.2 имеет регрессию для InternVL по видео-пути (баг #25227)

### Рекомендуемая версия для InternVL3.5

**vLLM 0.10.1** — подтверждённо рабочая для `OpenGVLab/InternVL3_5-8B` (оригинал, не HF).

Требования:
- `--trust-remote-code`
- `transformers >= 4.52.1`
- Без `video` в `--limit-mm-per-prompt` (только `{"image": N}`)

## Nominатрия: почему модели не запускаются

### InternVL3.5-8B-HF на vLLM 0.8.5
Образ 0.8.5 вышел в апреле 2025, до релиза InternVL3.5. Конфиг модели физически неизвестен этой версии.

### InternVL3.5-8B-HF на vLLM 0.10.2
HF-вариант использует `InternVLForConditionalGeneration`. Контрольная точка ожидает `start_image_token` и другие спец-токены, отсутствующие в `tokenizer_config.json`. Разрешение тайлов 384 vs 448 вызывает `ValueError` в `pixel_values`.

### Janus-Pro-7B (все версии vLLM)
Тип модели `multi_modality` не распознан vLLM. Нужен кастомный пакет `janus` от DeepSeek и запуск через Transformers (`AutoModelForCausalLM` + `VLChatProcessor`). Issue #12492 закрыт как дубликат.

**Решение:** исключить из vLLM-бенчмарка. Запуск только через Transformers/SGLang.

## Правила выбора формата весов

| Репозиторий | Архитектура | vLLM-совместимость |
|---|---|---|
| `OpenGVLab/InternVL3_5-8B` | `InternVLChatModel` | Да (0.10.1+, `--trust-remote-code`) |
| `OpenGVLab/InternVL3_5-8B-HF` | `InternVLForConditionalGeneration` | Проблемно (баг тайлов) |

**Правило:** для vLLM использовать оригинальные репозитории (не `-HF`), если модель имеет кастомную архитектуру.

## Запуск VLM на RTX 4090 (24 GB)

### Бюджет VRAM

| Размер модели | FP16 | AWQ/GPTQ (4-bit) | FP8 |
|---|---|---|---|
| 7-8B | ~16 GB | ~5-6 GB | ~8-9 GB |
| 14-15B | не влезает | ~9-10 GB | ~15 GB |
| 30-33B | не влезает | ~18-20 GB | не влезает |

### Команда запуска VLM (универсальная)

```bash
docker run -d --gpus '"device=1"' \
  --name llmbench-vllm-test \
  -p 8077:8077 \
  -v /opt/hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:<VERSION> \
  --model <HF_REPO> \
  --host 0.0.0.0 --port 8077 \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --served-model-name <NAME>
```

## Roadmap тестовых моделей

### Приоритет 1 — ближайший прогон

| Модель | Параметры | vLLM | Контекст | Заметки |
|---|---|---|---|---|
| InternVL3.5-8B (оригинал) | 8B | 0.10.1 | 128K | VLM, ViT+Qwen3, dynamic resolution |
| Molmo2-8B | 8B | 0.10.2 | 66K | VLM, AllenAI, Apache-2.0 |

### Приоритет 2 — требуют квантования

| Модель | Параметры | vLLM | Квант | Заметки |
|---|---|---|---|---|
| InternVL3.5-14B | 15B | 0.10.1+ | AWQ/FP8 | FP16 не влезет в 24 GB |
| InternVL3-14B | 15B | 0.8.5+ | AWQ/FP8 | Зрелая поддержка |
| Qwen3-VL-32B-Instruct | 33B | стабильно | Q4 | Тяжело, но реально |
| Qwen3-VL-30B-A3B | 31B (3B active) | стабильно | Q4 | MoE |

### Приоритет 3 — требуют nightly-vLLM

| Модель | Параметры | Заметки |
|---|---|---|
| Kimi-VL-A3B-Thinking | 16B (3B active) | MoE, 128K, reasoning |
| Qwen3.5-9B/4B/2B | 2-9B | Early-fusion multimodal |
| Qwen3.5-35B-A3B | 35B (3B active) | MoE, нужен квант |

### Приоритет 4 — специализированные (OCR)

| Модель | Параметры | Заметки |
|---|---|---|
| DeepSeek-OCR | 3B MoE | layout, таблицы, SMILES, ~100 языков |
| PaddleOCR-VL | 0.9B | OCR only, 109 языков |

### Исключены

| Модель | Причина |
|---|---|
| Janus-Pro-7B | Нет поддержки vLLM (model type `multi_modality`) |
| Qwen2.5-VL-7B | Уже протестирована (production) |
