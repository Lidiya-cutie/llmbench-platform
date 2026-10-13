#!/usr/bin/env python3
"""
Progress Tracker для интеграции новых моделей
Отслеживает выполнение чеклиста в реальном времени
"""

from pathlib import Path
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Any

class IntegrationTracker:
    """Отслеживание прогресса интеграции новых моделей"""

    def __init__(self, checklist_path: str = "NEW_MODELS_INTEGRATION_CHECKLIST.md"):
        self.checklist_path = Path(checklist_path)
        self.progress_file = Path("integration_progress.json")
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict[str, Any]:
        """Загрузка текущего прогресса"""
        if self.progress_file.exists():
            try:
                return json.loads(self.progress_file.read_text())
            except:
                return {"models": {}, "last_updated": None}
        return {"models": {}, "last_updated": None}

    def _save_progress(self):
        """Сохранение текущего прогресса"""
        self.progress["last_updated"] = datetime.now().isoformat()
        self.progress_file.write_text(json.dumps(self.progress, indent=2, ensure_ascii=False))

    def check_model_status(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Проверка статуса конкретной модели"""
        status = {
            "model_name": model_name,
            "checks": {},
            "overall_progress": 0.0,
            "last_check": datetime.now().isoformat(),
            "ready_for_testing": False
        }

        # Phase 0: Pre-flight
        preflight = self._check_preflight(model_config)
        status["checks"]["preflight"] = preflight

        # Phase 1: Setup
        setup = self._check_setup(model_config)
        status["checks"]["setup"] = setup

        # Phase 2: Server
        server = self._check_server(model_config)
        status["checks"]["server"] = server

        # Phase 3: API Test
        api = self._check_api(model_config)
        status["checks"]["api"] = api

        # Phase 4: Integration
        integration = self._check_integration(model_config)
        status["checks"]["integration"] = integration

        # Подсчет общего прогресса
        total_checks = 5
        completed_checks = sum([
            preflight["status"] == "completed",
            setup["status"] == "completed",
            server["status"] == "completed",
            api["status"] == "completed",
            integration["status"] == "completed"
        ])

        status["overall_progress"] = round(completed_checks / total_checks * 100, 1)
        status["ready_for_testing"] = all([
            server["status"] == "completed",
            api["status"] == "completed"
        ])

        return status

    def _check_preflight(self, config: Dict) -> Dict:
        """Phase 0: Предварительные проверки"""
        try:
            # GPU check
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                return {"status": "failed", "error": "GPU check failed", "details": result.stderr}

            free_vram = int(result.stdout.strip())

            # Disk check
            disk_result = subprocess.run(
                ["df", "-h", "/opt/data"],
                capture_output=True, text=True, timeout=10
            )

            return {
                "status": "completed" if free_vram > 15000 else "failed",
                "gpu_available": f"{free_vram} MiB",
                "recommendation": "Ready for setup" if free_vram > 15000 else "Insufficient VRAM"
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_setup(self, config: Dict) -> Dict:
        """Phase 1: Настройка окружения"""
        model_name = config.get("model_name", "")

        if "janus" in model_name.lower():
            model_path = Path("/opt/llm-data/my_train_models/deepseek_and_colpali/Janus/model2")
            port = 8099
        elif "internvl" in model_name.lower():
            # Для InternVL проверим несколько путей
            possible_paths = [
                Path("/opt/llm-data/internvl3_5/env"),
                Path("/opt/llm-data/internvl3_5"),
                Path.home() / ".cache/huggingface/hub"
            ]
            model_path = None
            for p in possible_paths:
                if p.exists():
                    model_path = p
                    break
            port = 8082
        else:
            return {"status": "skipped", "error": "Unknown model"}

        # Проверка файлов
        if model_name == "janus-pro-7b":
            if model_path.exists():
                bin_files = list(model_path.glob("*.bin"))
                return {
                    "status": "completed" if bin_files else "failed",
                    "model_files": len(bin_files),
                    "path": str(model_path)
                }

        elif model_name == "internvl3-5-8b":
            # Проверка venv
            venv_path = Path("/opt/llm-data/internvl3_5/env")
            if venv_path.exists():
                return {
                    "status": "completed",
                    "venv_ready": True,
                    "path": str(venv_path)
                }
            else:
                return {
                    "status": "pending",
                    "venv_ready": False,
                    "recommendation": "Run: cd /opt/llm-data/internvl3_5 && python3 -m venv env"
                }

        return {"status": "skipped", "note": "Model not recognized"}

    def _check_server(self, config: Dict) -> Dict:
        """Phase 2: Проверка сервера"""
        model_name = config.get("model_name", "")

        if "janus" in model_name.lower():
            port = 8099
        elif "internvl" in model_name.lower():
            port = 8082
        else:
            return {"status": "skipped", "error": "Unknown model"}

        # Проверка порта
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/health"],
                capture_output=True, text=True, timeout=5
            )

            # Проверка процесса
            ps_result = subprocess.run(
                ["ps", "aux"],
                capture_output=True, text=True, timeout=5
            )

            server_running = f"{port}" in ps_result.stdout

            if result.returncode == 0 or server_running:
                return {
                    "status": "completed",
                    "port": port,
                    "health": "OK" if result.returncode == 0 else "N/A",
                    "process_running": server_running
                }
            else:
                return {
                    "status": "pending",
                    "port": port,
                    "recommendation": f"Start server on port {port}"
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_api(self, config: Dict) -> Dict:
        """Phase 3: Проверка API"""
        model_name = config.get("model_name", "")

        if "janus" in model_name.lower():
            port = 8099
            model_name = "Janus-Pro-7B"
        elif "internvl" in model_name.lower():
            port = 8082
            model_name = "InternVL3.5-8B-HF"
        else:
            return {"status": "skipped", "error": "Unknown model"}

        try:
            # Тестовый запрос
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "2+2=?"}],
                "max_tokens": 10
            }

            import json
            result = subprocess.run(
                ["curl", "-X", "POST", f"http://localhost:{port}/v1/chat/completions",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(payload)],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0 and "4" in result.stdout:
                return {
                    "status": "completed",
                    "api_working": True,
                    "response": "Correct"
                }
            else:
                return {
                    "status": "pending",
                    "api_working": False,
                    "recommendation": "Check server logs"
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_integration(self, config: Dict) -> Dict:
        """Phase 4: Проверка интеграции"""
        model_name = config.get("model_name", "")

        # Проверка конфигурации
        config_yaml = Path("config/models.yaml")
        if not config_yaml.exists():
            return {"status": "pending", "recommendation": "Create config/models.yaml"}

        try:
            import yaml
            with open(config_yaml) as f:
                models_config = yaml.safe_load(f)

            if model_name in models_config.get("models", {}):
                return {
                    "status": "completed",
                    "config_exists": True,
                    "ready_for_benchmark": True
                }
            else:
                return {
                    "status": "pending",
                    "config_exists": False,
                    "recommendation": f"Add {model_name} to config/models.yaml"
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate_progress_report(self) -> str:
        """Генерация отчета о прогрессе"""
        report = ["# 📊 Progress Report: New Models Integration", ""]

        report.append(f"**Updated:** {self.progress.get('last_updated', 'Never')}")
        report.append("")

        # Проверяем обе модели
        models_config = [
            {"model_name": "janus-pro-7b", "name": "Janus-Pro-7B", "type": "Multimodal"},
            {"model_name": "internvl3-5-8b", "name": "InternVL3.5-8B", "type": "Multimodal"}
        ]

        for model_config in models_config:
            status = self.check_model_status(model_config["model_name"], model_config)

            # Обновляем кэш
            self.progress["models"][model_config["model_name"]] = status
            self._save_progress()

            # Добавляем в отчет
            progress_emoji = "✅" if status["overall_progress"] == 100 else "🔄" if status["overall_progress"] > 0 else "❌"

            report.append(f"## {progress_emoji} {model_config['name']} ({status['overall_progress']}%)")
            report.append(f"- **Status:** {status.get('checks', {}).get('server', {}).get('status', 'unknown')}")
            report.append(f"- **API:** {status.get('checks', {}).get('api', {}).get('api_working', 'unknown')}")
            report.append(f"- **Integration:** {status.get('checks', {}).get('integration', {}).get('config_exists', 'unknown')}")
            report.append(f"")

            # Рекомендации
            recommendations = []
            for phase_name, phase_data in status.get("checks", {}).items():
                if phase_data.get("status") == "pending" and "recommendation" in phase_data:
                    recommendations.append(f"- {phase_data['recommendation']}")

            if recommendations:
                report.append("**Next Steps:**")
                for rec in recommendations:
                    report.append(f"  {rec}")
                report.append("")

        report.append("---")
        report.append("**Ready for Full Testing:**")
        ready_models = [
            k for k, v in self.progress.get("models", {}).items()
            if v.get("ready_for_testing", False)
        ]

        if ready_models:
            report.append(", ".join(ready_models))
        else:
            report.append("None (need setup)")

        return "\n".join(report)


def main():
    tracker = IntegrationTracker()
    print(tracker.generate_progress_report())


if __name__ == "__main__":
    main()