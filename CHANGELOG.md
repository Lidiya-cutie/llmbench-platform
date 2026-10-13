# История изменений

Формат: [SemVer](https://semver.org/lang/ru/) (MAJOR.MINOR.PATCH).

---

## [0.2.0] — 2026-08-12

### Добавлено
- Реструктуризация документации по стандартам ML-команды
- `glossary.md` — глоссарий терминов бенчмаркинга и инфраструктуры
- `docs/installation_guide.md` — руководство по установке
- `docs/deployment_guide.md` — руководство по развёртыванию
- `docs/developer_guide.md` — руководство разработчика
- `docs/architecture.md` — архитектурный обзор системы
- `docs/troubleshooting.md` — устранение неполадок
- `CHANGELOG.md` — история изменений

### Изменено
- `README.md` — полная переработка по шаблону "Представление проекта"
- Структура документации приведена к стандарту doc-guideline

---

## [0.1.0] — 2026-07-15

### Добавлено
- Базовая платформа оценки LLM на RTX 4090
- Модули: `hwmon`, `perf`, `micro_eval`, `lmeval`, `cost`, `modelcard`
- Интеграция с lm-evaluation-harness 0.4.12
- Карточки моделей (JSON + Markdown)
- Дашборд (HTML)
- Скрипты: `evaluate_model.py`, `compare_models.py`, `launch_model.py`
- Конфигурации: `decision.yaml`, `vision_domain.yaml`
- CI/CD: GitHub Actions (lint + test)
- Pre-commit hooks: black, ruff, mypy, pytest, trufflehog
- Тесты: test_pure (38), test_internal (32), test_decision (18)

### Известные ограничения
- HumanEval: баг в lm-eval 0.4.12 (лишняя скобка в `build_predictions`) — требуется локальный патч
- Прокси Gemma: race condition при concurrent-доступе к Triton
- GPU0: критическое переполнение при работе 27B AWQ моделей
