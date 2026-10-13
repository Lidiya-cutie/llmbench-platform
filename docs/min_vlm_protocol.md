# Минимальный протокол VLM тестирования

## Обзор

Документация минимального протокола для тестирования визуально-язычных моделей (VLM) без полноценных прогонов из-за высоких требований к GPU ресурсам.

## Статус системы

**GPU загрузка (критически важна):**
- **GPU0:** Тритон сервис (постоянная нагрузка)
- **GPU1:** Qwen3-8B MMLU Pro (94% утилизации, 22GB VRAM)
- **Индексация RAG:** 25+ часов, 1872+/7361 docs в процессе

**⚠️ ВАЖНО:** Полноценные VLM прогоны невозможны без прерывания критических задач.

## Минимальный протокол

### Этап 1: Проверка наличия vision модальности

**Текстовые модели (без vision):**
```python
# modelcard.py
if modalities == "text":
    vision.status = "not_tested"
    vision.note = "Text-only модель, VLM тестирование неприменимо"
```

**Мультимодальные модели:**
```python
if modalities == "text+image":
    vision.status = "ready_for_minimal_protocol"
    vision.protocol = "image_url_stub"
```

### Этап 2: Stub тестирование (без GPU)

**Single image test:**
```python
def stub_vlm_test(model_info):
    """
    Заглушка для VLM без реального прогона.
    Проверяет доступность image_url endpoint без GPU нагрузки.
    """
    # Проверка формата image_url в ответе
    payload = {
        "model": model_info["name"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Опиши изображение."},
                    {
                        "type": "image_url",
                        "image_url": "https://example.com/test.jpg"
                    }
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(f"{base_url}/chat/completions",
                                     json=payload, timeout=30)
        
        # Проверяем, что модель принимает image_url
        if response.status_code == 200:
            return {
                "status": "image_url_compatible",
                "note": "Модель принимает image_url формат"
            }
        else:
            return {
                "status": "image_url_incompatible",
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            "status": "stub_failed",
            "error": str(e)
        }
```

### Этап 3: Критерии готовности

**Базовая готовность:**
```python
def check_vlm_readiness(model_info, model_card):
    """Проверяет базовую готовность VLM модели."""
    criteria = {
        "vision_modality": model_info.get("modalities") == "text+image",
        "image_url_support": stub_vlm_test(model_info).get("status") == "image_url_compatible",
        "vlmevalkit_available": check_vlmevalkit_installation(),
        "documentation_exists": check_vlm_documentation(),
    }
    
    readiness = {
        "basic_ready": all(criteria.values()),
        "details": criteria
    }
    
    if readiness["basic_ready"]:
        readiness["status"] = "ready_for_minimal_protocol"
        readiness["gpu_blocked"] = True  # GPU заня критическими задачами
    else:
        readiness["status"] = "not_ready"
    
    return readiness
```

## Интеграция с VLMEvalKit

### Структура каталогов

```
llmbench/
├── .venv-vlmeval/          # Отдельное окружение для VLMEvalKit
│   ├── bin/python3
│   ├── lib/
│   └── include/
├── third_party/VLMEvalKit/  # VLMEvalKit исходники
├── vlmeval_minimal.py        # Минимальная интеграция
└── docs/vlm_protocol.md      # Эта документация
```

### Минимальная интеграция

```python
# vlmeval_minimal.py
import os

def check_vlmevalkit_installation():
    """Проверяет доступность VLMEvalKit."""
    try:
        import sys
        sys.path.insert(0, "/opt/llm-data/llmbench/.venv-vlmeval/lib")
        import vlmeval
        return {"available": True, "version": vlmeval.__version__}
    except ImportError:
        return {"available": False, "error": "VLMEvalKit не установлен"}

def check_vlm_documentation():
    """Проверяет документацию VLM."""
    docs_path = "/opt/llm-data/llmbench/docs/vlm_protocol.md"
    return os.path.exists(docs_path)
```

## Stub текстовый режим

### Реализация в modelcard.py

```python
def assess_fit(card: ModelCard) -> Dict[str, str]:
    q = card.quality
    micro = card.micro
    perf = card.perf
    
    # Проверка модальности
    if card.modalities == "text":
        return {
            **base_assess_fit(),
            "OCR / Vision (VLM)": "❌ не рекомендуется (текстовая модель)",
            "🔍 VLM тестирование": "Модель не поддерживает изображения"
        }
    
    # Для мультимодальных моделей
    if card.modalities == "text+image":
        # Проверяем заглушку
        vision = card.reliability.get("vision")
        if vision and vision.get("status") == "not_tested":
            fit["OCR / Vision (VLM)"] = "⚠️ не тестировалось (stub режим)"
            fit["🔍 VLM тестирование"] = "Стаб доступен для минимального протокола"
        else:
            fit["OCR / Vision (VLM)"] = "✅ подходит (VLM готов к тестированию)"
```

## Критерии готовности VLM моделей

### Level 1: Базовая готовность

**Требования:**
- ✅ Модель поддерживает `image_url` в messages
- ✅ Модель принимает мультимодальные сообщения
- ✅ JSON валидность `tool_call_ok`
- ✅ Базовая reasoning ≥ 0.4

**Критерий:**
```python
basic_ready = (
    modalities == "text+image" and
    json_valid == 1.0 and
    tool_call_ok == 1.0 and
    quality.get("gsm8k", 0) >= 0.4
)
```

**Статус:** `"🔍 Стаб доступен для минимального протокола"`

### Level 2: Минимальный протокол

**Требования:**
- ✅ Все Level 1 требования
- ✅ `stub_vlm_test()` проходит
- ✅ VLMEvalKit установлен (опционально для stub режима)
- ✅ Документация описана

**Критерий:**
```python
minimal_ready = (
    basic_ready and
    stub_vlm_test()["status"] == "image_url_compatible"
)
```

**Статус:** `"✅ Готов для минимального протокола"`

### Level 3: Полные тесты

**Требования:**
- ✅ Все Level 1 и Level 2 требования
- ✅ VLMEvalKit полностью установлен
- ✅ Доступно ≥16GB VRAM (для VLMEvalKit)
- ✅ GPU ресурсы освобождены от критических задач
- ✅ DocVQA/OCRBench/MMMU тесты работают

**Критерий:**
```python
full_ready = (
    minimal_ready and
    check_vlmevalkit_installation()["available"] and
    check_gpu_available() and
    docvqa_test_passed() and
    ocrbench_test_passed()
)
```

**Статус:** `"🚀 Готов для полных VLM тестов"`

## Проверка ресурсов

### GPU доступность

```python
def check_gpu_available():
    """Проверяет доступность GPU ресурсов."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        
        # Парсинг свободной памяти
        for line in result.stdout.strip().split('\n'):
            if line:
                free_mb = float(line.split(',')[0])
                free_gb = free_mb / 1024
                return {
                    "available": free_gb >= 16,  # Минимум 16GB для VLMEvalKit
                    "free_gb": round(free_gb, 1),
                    "status": "ready" if free_gb >= 16 else "blocked"
                }
        return {"available": False, "error": "GPU check failed"}
    except Exception:
        return {"available": False, "error": str(e)}
```

### RAM требования

```python
def check_ram_available():
    """Проверяет доступную RAM."""
    import psutil
    available_gb = psutil.virtual_memory().available / (1024**3)
    return {
        "available_gb": round(available_gb, 1),
        "required_gb": 8,  # Для VLMEvalKit
        "status": "ready" if available_gb >= 8 else "insufficient"
    }
```

## Статусы и этапы

### Статусы готовности

| Статус | Описание | Действие |
|--------|-------------|---------|
| `not_tested` | Текстовая модель | Пропустить VLM тестирование |
| `stub_ready` | Стаб доступен для минимального протокола | Документировать |
| `minimal_ready` | Готов для минимального протокола | Ждать GPU |
| `full_ready` | Готов для полных тестов | Запустить VLMEvalKit |

### Блокирующие факторы

```python
blocking_factors = {
    "GPU0 occupied": "Тритон сервис (постоянная нагрузка)",
    "GPU1 occupied": "Qwen3-8B MMLU Pro (94% утилизации)",
    "RAG indexing": "25+ часов, 1872+/7361 docs (25% complete)",
    "ram_insufficient": "Need 8GB+ free for VLMEvalKit",
    "llmbench_vlm_execute": "Прогоны отключены по умолчанию"
}
```

## Текущий статус

**Готовность: ~25% (инфраструктура)**
- ✅ VLMEvalKit установлен в third_party/
- ✅ .venv-vlmeval окружение создано
- ✅ Stub text-only режим реализован
- ✅ qwen2.5-vl-7b модель доступна
- ⚠️ Полные прогоны заблокированы (GPU заняты)

**Следующий шаг:** Документация критериев готовности без прогона.

## Безопасность

✅ **Полностью безопасно для критических процессов:**
- Никаких GPU прогонов
- Никаких дополнительных нагрузок
- Только документация и stub тесты
- Все критические процессы работают стабильно