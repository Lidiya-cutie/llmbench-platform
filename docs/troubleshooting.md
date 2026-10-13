# Устранение неполадок

## GPU и аппаратные проблемы

### CUDA out of memory (OOM)

**Симптомы:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```
Или vLLM падает при загрузке модели.

**Причины и решения:**

| Причина | Решение |
|---|---|
| Модель слишком большая для VRAM | Использовать квантизацию (AWQ, GPTQ, GGUF) |
| KV-cache переполняет VRAM | Уменьшить `--max-model-len` (например, с 8192 до 4096) |
| Другой процесс занимает GPU | `nvidia-smi`, остановить лишние процессы |
| `gpu_memory_utilization` слишком высоко | Уменьшить с 0.90 до 0.70-0.80 |

**Диагностика:**

```bash
# Что занимает GPU
nvidia-smi
fuser -v /dev/nvidia*

# Свободная VRAM
nvidia-smi --query-gpu=memory.free --format=csv
```

### GPU не определяется

**Симптомы:** `torch.cuda.is_available()` возвращает `False`

**Решения:**

1. Проверить драйвер: `nvidia-smi`
2. Переустановить PyTorch с правильной версией CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
3. Проверить `CUDA_VISIBLE_DEVICES`:
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   # Если пусто — все GPU видны
   # Если "1" — только GPU1
   ```

### Перегрев GPU

**Симптомы:** thermal throttling, температура >83C

**Решения:**
- Проверить охлаждение корпуса
- Установить `nvidia-smi -pl 300` (ограничение мощности до 300W)
- Уменьшить `gpu_memory_utilization`

---

## vLLM

### vLLM не запускается

**Симптомы:** процесс падает в течение 30 секунд после запуска

**Диагностика:**

```bash
# Логи
tail -100 logs/vllm_*.log

# Проверить порты
ss -tulpn | grep 8077

# Проверить Docker
docker ps -a | grep vllm
docker logs <container_name>
```

**Частые причины:**

| Причина | Решение |
|---|---|
| Порт занят | `python scripts/launch_model.py --port 8078` |
| Недостаточно VRAM | См. CUDA OOM выше |
| Неверный путь к модели | Проверить путь в `launch_cmd` |
| Несовместимый формат | Проверить `--dtype` и квантизацию |

### vLLM краш при длинных промптах

**Причина:** `max_model_len` недостаточен для prompt + generation

**Решение:**
```bash
# Увеличить max_model_len
vllm serve ... --max-model-len 16384

# Или ограничить длину prompt в оценке
python scripts/evaluate_model.py ... --perf-max-tokens 256
```

---

## lm-evaluation-harness

### HumanEval возвращает 0%

**Причина:** Баг в lm-eval 0.4.12 — лишняя закрывающая скобка в `tasks/humaneval/utils.py:build_predictions()`.

**Патч:**

```bash
# Найти файл
HUMANEVAL_UTILS=$(python -c "import lm_eval; print(lm_eval.__path__[0])")/tasks/humaneval/utils.py

# Применить патч
sed -i 's/zip(resps, docs)]]/zip(resps, docs)]/' "$HUMANEVAL_UTILS"

# Проверить
grep "zip(resps, docs)" "$HUMANEVAL_UTILS"
# Должно быть: ...for resp, doc in zip(resps, docs)]
```

### lm-eval процесс умирает "тихо"

**Симптомы:** процесс завершается без трейсбэка или с BrokenPipeError

**Причины:**

1. **BrokenPipe при потере соединения с inference-сервером** — lm-eval не обрабатывает `outputs=None` после исчерпания ретраев
2. **OOM killer** — процесс убит ядром Linux

**Решения:**

- Использовать checkpoint/resume:
  ```bash
  python scripts/run_with_ckpt.py --resume --model-key MODEL
  ```
- Уменьшить `num_concurrent` до 1
- Увеличить таймаут: `--timeout 600`
- Мониторить OOM: `dmesg | grep -i "out of memory"`

### Ошибки парсинга результатов

**Симптомы:** `quality` в карточке пустой после завершения lm-eval

**Диагностика:**

```bash
# Проверить сырые результаты
ls out/lmeval_*_queue/
cat out/lmeval_*_queue/results.json | python3 -m json.tool | head -30
```

**Решение:**

```bash
# Ручной merge
python scripts/merge_lmeval_into_card.py \
  --card out/card_MODEL.json \
  --lmeval-root out/lmeval_MODEL_queue
```

---

## Дашборд

### Дашборд не отображает данные

**Симптомы:** `out/dashboard_v2.html` пустой или показывает не все модели

**Диагностика:**

```bash
# Проверить наличие карточек
ls out/card_*.json

# Проверить валидность карточек
python3 -m llmbench.validate --all --out-dir out

# Проверить JSON структуру
python3 -c "
import json, glob
for f in sorted(glob.glob('out/card_*.json')):
    d = json.load(open(f))
    print(f'{f}: quality={bool(d.get(\"quality\"))}, perf={bool(d.get(\"perf\"))}')
"
```

**Решение:**

```bash
# Регенерация дашборда
python scripts/auto_update_all_cards.py --out-dir out
python scripts/generate_dashboard_v2.py
```

### Карточка модели некорректна

**Симптомы:** метрики равны 0 или отсутствуют

**Решение:**

```bash
# Валидация конкретной карточки
python3 -m llmbench.validate --card out/card_MODEL.json

# Исправление через merge
python scripts/merge_lmeval_into_card.py \
  --card out/card_MODEL.json \
  --lmeval-root out/lmeval_MODEL_queue

# Регенерация Markdown
python scripts/auto_generate_cards.py
```

---

## Сеть и доступ

### Прокси / GitLab недоступен

**Симптомы:** `git clone` или `git push` к `git.example.com` не работает

**Решение:** GitLab доступен только через прокси:

```bash
# Настройка git proxy
git config --global http.https://git.example.com.proxy socks5h://127.0.0.1:10810

# Проверка прокси
curl -x socks5h://127.0.0.1:10810 https://git.example.com
```

### Docker нет доступа

**Симптомы:** `docker ps` возвращает permission denied

**Решение:**

```bash
# Добавить пользователя в группу docker
sudo usermod -aG docker $USER
newgrp docker

# Или использовать sudo
sudo docker ps
```

### HuggingFace download медленный / обрывы

**Решение:**

```bash
# Использовать HF Mirror
export HF_ENDPOINT=https://hf-mirror.com

# Или скачать вручную
huggingface-cli download org/model --local-dir /path/to/model
```

---

## Оркестрация

### Оркестратор не запускает модель на свободном GPU

**Диагностика:**

```bash
# Статус GPU
python scripts/orchestrator.py --status

# Проверить, что GPU действительно свободен
nvidia-smi
```

**Решение:** Убедиться, что `CUDA_VISIBLE_DEVICES` не ограничивает видимость GPU для процесса.

### Concurrent оценка вызывает конфликты

**Причина:** Два процесса оценки обращаются к одному inference-серверу

**Решение:** Каждый процесс оценки должен работать со своим эндпоинтом (разный порт). Использовать оркестратор для координации.
