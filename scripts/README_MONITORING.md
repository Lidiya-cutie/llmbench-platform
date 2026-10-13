# Мониторинг завершения glm-4-9b-chat прогона

## Быстрая проверка статуса

```bash
/opt/llmbench/scripts/check_glm_status.sh
```

Показывает:
- Статус прогона (АКТИВЕН/ЗАВЕРШЕН)
- Текущий прогресс (N/25026)
- Загрузку GPU
- Оставшееся время (если активен)

## Детальный мониторинг

```bash
/opt/llmbench/scripts/monitor_glm_progress.sh
```

Полная информация с расчетом времени.

## Автоматическое ожидание

```bash
/opt/llmbench/scripts/wait_for_glm_complete.sh
```

Скрипт ждет завершения прогона и логирует события. После завершения выведет инструкции для следующих шагов.

## Ручной мониторинг

**Процессы:**
```bash
ps aux | grep -E "(glm-4-9b|lm_eval)" | grep -v grep
```

**Лог в реальном времени:**
```bash
tail -f /opt/llmbench/logs/eval_glm-4-9b-chat.log
```

**Прогресс из лога:**
```bash
tail -1 /opt/llmbench/logs/eval_glm-4-9b-chat.log | grep -oP "\d+/25026"
```

## Индикаторы завершения

1. **Процесс завершен:**
   ```bash
   ! pgrep -f "lm_eval.*glm-4-9b-chat"
   ```

2. **Карточка обновлена:**
   ```bash
   ls -la /opt/llmbench/out/card_glm-4-9b-chat.json
   # Время обновления должно быть недавним
   ```

3. **Quality поле заполнено:**
   ```bash
   cat /opt/llmbench/out/card_glm-4-9b-chat.json | grep -A 10 '"quality"'
   # Должны быть метрики вместо {}
   ```

4. **Нет ошибок в логе:**
   ```bash
   tail -20 /opt/llmbench/logs/eval_glm-4-9b-chat.log | grep -i "error\|fatal"
   # Должно быть пусто
   ```

## Критерии успеха

- ✅ Процесс lm_eval завершен
- ✅ Карточка обновлена (время < 10 мин назад)
- ✅ Quality поле содержит метрики (не `{}`)
- ✅ В логах нет ERROR/FATAL

## Следующие шаги после завершения

1. Остановить vLLM:
   ```bash
   docker rm -f vllm-bench
   ```

2. Запустить local-27b ternary на GPU1:
   ```bash
   export LOCAL27B=/opt/models/local-27b
   export LD_LIBRARY_PATH="$LOCAL27B/bin/cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   CUDA_VISIBLE_DEVICES=1 "$LOCAL27B/bin/cuda/llama-server" \
     -m "$LOCAL27B/models/ternary-gguf/27B/Ternary-local-27b-Q2_0.gguf" \
     --mmproj "$LOCAL27B/models/ternary-gguf/27B/Ternary-local-27b-mmproj-Q8_0.gguf" \
     --host 127.0.0.1 --port 8078 -ngl 99 -fa on -c 8192 -np 1 \
     --temp 0.7 --top-p 0.95 --top-k 20 --min-p 0 --jinja \
     --reasoning off \
     >>/opt/llmbench/logs/local-27b_ternary_gpu1.log 2>&1 &
   ```

3. Запустить evaluate для local-27b:
   ```bash
   cd /opt/llmbench
   .venv/bin/python scripts/evaluate_model.py \
     --model-key local-27b-27b-ternary \
     --base-url http://127.0.0.1:8078/v1 \
     --served-model Ternary-local-27b-Q2_0.gguf \
     --suite full \
     --context-scaling \
     --perf-n 12 \
     --perf-max-tokens 256 \
     --no-thinking-body \
     --out-dir out
   ```

## Текущий статус (16 июля 09:10)

- glm-4-9b-chat: АКТИВЕН (~200+/25026 запросов, ~1%)
- Оставшееся время: ~28 часов
- GPU0: 96% загрузка
- GPU1: свободен для local-27b
