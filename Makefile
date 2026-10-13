# Makefile — llmbench production entry point
#
# Usage:
#   make setup      — создать venv, установить зависимости
#   make serve      — подустить vLLM через docker compose
#   make eval       — запустить бенчмарк (MODEL=qwen3-8b TASKS=mmlu_pro,gsm8k)
#   make vision     — запустить DocVQA/OCRBench
#   make dashboard  — подустить HTTP сервер дашборда
#   make test       — прогнать pytest
#   make lint       — black + ruff
#   make health     — проверить здоровье системы
#   make cards      — сгенерировать карточки моделей
#   make clean      — очистить кэш

SHELL := /bin/bash
.DEFAULT_GOAL := help

# === Переменные ===
LLMBENCH_DIR ?= /opt/llm-data/llmbench
VENV ?= $(LLMBENCH_DIR)/.venv
PY ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
DOCKER_COMPOSE ?= docker compose
MODEL ?= qwen3-8b
TASKS ?= mmlu_pro,gsm8k,minerva_math,gpqa,humaneval,mbpp,ifeval
BASE_URL ?= http://127.0.0.1:8070
SERVED_MODEL ?= Qwen3-8B
DASHBOARD_PORT ?= 8899

# === Phony targets ===
.PHONY: help setup serve serve-vl stop serve-dashboard eval vision cards test lint health clean

help: ## Показать справку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Создать venv и установить зависимости
	$(info -> Создание venv $(VENV))
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip wheel
	$(PIP) install -e ".[dev]"
	$(info -> Установка lm-eval-harness)
	$(PIP) install lm_eval==0.4.12
	$(info -> Готово. Активация: source $(VENV)/bin/activate)

serve: ## Поднять vLLM на GPU0 (текстовые модели)
	$(info -> Запуск vLLM на GPU0 через docker compose)
	$(DOCKER_COMPOSE) --env-file .env up -d vllm-gpu0
	@echo "Ожидание health..."
	@for i in $$(seq 1 30); do \
		curl -sf http://localhost:$(VLLM_PORT_GPU0:-8070)/health && echo " OK" && break; \
		echo -n "."; sleep 5; \
	done

serve-vl: ## Поднять vLLM на GPU1 (Vision модели)
	$(info -> Запуск vLLM VL на GPU1 через docker compose)
	$(DOCKER_COMPOSE) --env-file .env up -d vllm-gpu1

stop: ## Остановить все контейнеры llmbench
	$(DOCKER_COMPOSE) down

serve-dashboard: ## Поднять HTTP сервер дашборда
	$(info -> Дашборд на порту $(DASHBOARD_PORT))
	cd $(LLMBENCH_DIR)/out && python3 -m http.server $(DASHBOARD_PORT)

eval: ## Запустить бенчмарк (MODEL=qwen3-8b TASKS=mmlu_pro,gsm8k)
	$(info -> Оценка модели $(MODEL), задачи: $(TASKS))
	HF_ALLOW_CODE_EVAL=1 $(PY) -m lm_eval \
		--model local-chat-completions \
		--model_args "model=$(SERVED_MODEL),base_url=$(BASE_URL)/v1/chat/completions,num_concurrent=1" \
		--tasks $(TASKS) \
		--output_path $(LLMBENCH_DIR)/out/lmeval_$(MODEL)_queue \
		--log_samples

vision: ## Запустить DocVQA/OCRBench (Vision)
	$(info -> Vision оценка через VLMEvalKit)
	LLMBENCH_VL_EXECUTE=1 \
	MODEL_CARD=qwen2.5-vl-7b \
	BASE_URL=http://127.0.0.1:8077 \
	SERVED_MODEL=Qwen2.5-VL-7B-Instruct \
	bash $(LLMBENCH_DIR)/scripts/run_vlm_docvqa_ocr.sh

cards: ## Сгенерировать карточки моделей
	$(info -> Генерация карточек)
	$(PY) $(LLMBENCH_DIR)/scripts/auto_generate_cards.py

test: ## Прогнать pytest
	$(VENV)/bin/pytest $(LLMBENCH_DIR)/tests/ -v --tb=short

lint: ## black + ruff
	$(VENV)/bin/black --check $(LLMBENCH_DIR)/llmbench/ $(LLMBENCH_DIR)/scripts/
	$(VENV)/bin/ruff check $(LLMBENCH_DIR)/llmbench/ $(LLMBENCH_DIR)/scripts/

health: ## Проверить здоровье системы
	$(PY) $(LLMBENCH_DIR)/scripts/healthcheck.py

clean: ## Очистить кэш
	find $(LLMBENCH_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find $(LLMBENCH_DIR) -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(LLMBENCH_DIR)/out/lmeval_*/samples 2>/dev/null || true
