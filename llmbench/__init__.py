"""llmbench-4090: минимальная, но рабочая платформа оценки локальных LLM на RTX 4090."""
from .hwmon import HardwareMonitor, HwSummary
from .perf import run_perf, run_context_scaling, PerfResult
from .micro_eval import run_micro_eval, MicroResult
from .lmeval import run_lm_eval, parse_results_file, LmEvalResult
from .cost import estimate_cost, CostInputs, CostResult
from .modelcard import ModelCard, render_markdown, finalize, assess_fit, build_conclusion
from .health import check_openai_endpoint, check_model_loaded, wait_ready, check_generate, HealthStatus

__all__ = [
    "HardwareMonitor", "HwSummary",
    "run_perf", "run_context_scaling", "PerfResult",
    "run_micro_eval", "MicroResult",
    "run_lm_eval", "parse_results_file", "LmEvalResult",
    "estimate_cost", "CostInputs", "CostResult",
    "ModelCard", "render_markdown", "finalize", "assess_fit", "build_conclusion",
    "check_openai_endpoint", "check_model_loaded", "wait_ready", "check_generate", "HealthStatus",
]
