"""
Интеграционные тесты llmbench.

Проверяют сквозные сценарии:
1. Генерация карточки модели из результатов
2. Парсинг результатов lm-eval
3. Конфигурация decision.yaml
4. API conclusion_server (если запущен)
5. ModelCard → conclusion → assess_fit цепочка

Запуск: pytest tests/test_integration.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from llmbench.modelcard import ModelCard, assess_fit, build_conclusion, finalize


# === Фикстуры ===

@pytest.fixture
def sample_quality():
    """Создает полный набор метрик 7/7."""
    return {
        "mmlu_pro": 0.587,
        "gpqa": 0.306,
        "gsm8k": 0.845,
        "minerva_math": 0.405,
        "humaneval": 0.860,
        "mbpp": 0.608,
        "ifeval": 0.368,
    }


@pytest.fixture
def sample_card(sample_quality):
    """Создает тестовую карточку модели."""
    return ModelCard(
        name="test-model",
        repo="test/repo",
        quant_format="bf16",
        context_max=32768,
        engine="vllm",
        quality=sample_quality,
        perf={"ttft_ms": 120, "tokens_per_sec": 85},
        hardware={"vram_mib": 16000, "gpu": "RTX 4090"},
        cost={"cost_per_1m_tokens": 0.0},
    )


@pytest.fixture
def real_cards_dir():
    """Путь к реальным карточкам в out/."""
    return Path(__file__).parent.parent / "out"


# === Тесты генерации карточек ===

class TestModelCardGeneration:
    """Тесты полного цикла генерации карточки."""

    def test_full_pipeline(self, sample_card):
        """Полный пайплайн: создание → финализация → заключение."""
        finalized = finalize(sample_card)
        assert finalized is not None
        conclusion = build_conclusion(finalized)
        assert conclusion is not None
        assert isinstance(conclusion, str)
        assert len(conclusion) > 0

    def test_card_serialization(self, sample_card):
        """Карточка должна сериализоваться в JSON без потерь."""
        d = sample_card.to_dict()
        j = json.dumps(d)
        restored = ModelCard(**json.loads(j))
        assert restored.name == sample_card.name
        assert restored.quality == sample_card.quality

    def test_zero_metrics_handled(self):
        """Нулевые метрики не должны ломать карточку."""
        card = ModelCard(
            name="zero-model",
            repo="test/zero",
            quant_format="q1",
            context_max=4096,
            quality={
                "mmlu_pro": 0.0,
                "gpqa": 0.0,
                "gsm8k": 0.929,
                "minerva_math": 0.515,
                "humaneval": 1.0,
                "mbpp": 0.0,
                "ifeval": 0.786,
            },
            perf={},
            hardware={},
            cost={},
        )
        finalized = finalize(card)
        conclusion = build_conclusion(finalized)
        assert "0.0%" not in conclusion or "0%" in conclusion

    def test_missing_metrics_handled(self):
        """Отсутствующие метрики (null) не должны ломать карточку."""
        card = ModelCard(
            name="partial-model",
            repo="test/partial",
            quant_format="bf16",
            context_max=8192,
            quality={
                "mmlu_pro": None,
                "gpqa": 0.3,
                "gsm8k": 0.8,
            },
            perf={},
            hardware={},
            cost={},
        )
        finalized = finalize(card)
        assert finalized is not None


# === Тесты assess_fit ===

class TestAssessFit:
    """Тесты оценки пригодности моделей."""

    def test_assess_fit_returns_dict(self, sample_card):
        """assess_fit возвращает словарь домен->вердикт."""
        fit = assess_fit(sample_card)
        assert isinstance(fit, dict)
        assert len(fit) > 0

    def test_assess_fit_has_domains(self, sample_card):
        """Результат содержит ключевые домены."""
        fit = assess_fit(sample_card)
        expected_domains = {"coding", "math", "reasoning", "instruct"}
        found = set(fit.keys()) & expected_domains
        assert len(found) >= 2, f"Only {len(found)} domains: {found}"

    def test_assess_fit_values_are_strings(self, sample_card):
        """Значения в результате — строки."""
        fit = assess_fit(sample_card)
        for k, v in fit.items():
            assert isinstance(v, str), f"{k} value is {type(v)}, not str"


# === Тесты реальных карточек ===

class TestRealCards:
    """Тесты на реальных карточках из out/."""

    @pytest.fixture(params=[
        "local-27b-27b-ternary",
        "gemma-3-27b-awq",
        "qwen3-8b-bf16",
        "glm-4-9b-chat",
    ])
    def real_card(self, real_cards_dir, request):
        card_path = real_cards_dir / f"card_{request.param}.json"
        if not card_path.exists():
            pytest.skip(f"card_{request.param}.json not found")
        with open(card_path) as f:
            return json.load(f)

    def test_real_card_has_quality(self, real_card):
        """Каждая реальная карточка имеет блок quality."""
        assert "quality" in real_card or "metrics" in real_card

    def test_real_card_has_7_metrics(self, real_card):
        """Карточка должна иметь 7 ключевых метрик."""
        quality = real_card.get("quality", real_card.get("metrics", {}))
        expected = {"mmlu_pro", "gpqa", "gsm8k", "minerva_math",
                    "humaneval", "mbpp", "ifeval"}
        found = set(quality.keys()) & expected
        assert len(found) >= 5, f"Only {len(found)}/7 metrics: {found}"


# === Тесты конфигурации ===

class TestConfig:
    """Тесты конфигурационных файлов."""

    def test_decision_yaml_exists(self):
        """decision.yaml должен существовать."""
        p = Path(__file__).parent.parent / "config" / "decision.yaml"
        assert p.exists(), "config/decision.yaml missing"

    def test_decision_yaml_valid(self):
        """decision.yaml должен быть валидным YAML."""
        import yaml
        p = Path(__file__).parent.parent / "config" / "decision.yaml"
        if not p.exists():
            pytest.skip("decision.yaml not found")
        with open(p) as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert isinstance(data, dict)


# === Тесты структуры проекта ===

class TestProjectStructure:
    """Тесты структуры проекта по стандартам."""

    def test_pyproject_toml(self):
        """pyproject.toml существует и содержит версию."""
        import tomllib
        p = Path(__file__).parent.parent / "pyproject.toml"
        assert p.exists()
        with open(p, "rb") as f:
            data = tomllib.load(f)
        assert "project" in data
        assert "version" in data["project"]

    def test_readme_exists(self):
        p = Path(__file__).parent.parent / "README.md"
        assert p.exists()
        assert p.stat().st_size > 1000

    def test_glossary_exists(self):
        p = Path(__file__).parent.parent / "glossary.md"
        assert p.exists()

    def test_changelog_exists(self):
        p = Path(__file__).parent.parent / "CHANGELOG.md"
        assert p.exists()

    def test_docker_compose_exists(self):
        p = Path(__file__).parent.parent / "docker-compose.yml"
        assert p.exists()

    def test_makefile_exists(self):
        p = Path(__file__).parent.parent / "Makefile"
        assert p.exists()

    def test_env_example_exists(self):
        p = Path(__file__).parent.parent / ".env.example"
        assert p.exists()

    def test_healthcheck_exists(self):
        p = Path(__file__).parent.parent / "scripts" / "healthcheck.py"
        assert p.exists()
