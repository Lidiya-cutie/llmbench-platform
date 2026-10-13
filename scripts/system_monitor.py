#!/usr/bin/env python3
"""
System Health Monitor - Безопасный мониторинг состояния системы
Проверяет доступность моделей без прерывания активных процессов
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class SystemHealthMonitor:
    """Мониторинг системы без прерывания процессов"""

    def __init__(self, base_dir: str = "/opt/llm-data/llmbench"):
        self.base_dir = Path(base_dir)
        self.models_config = self._load_models_config()

    def _load_models_config(self) -> Dict[str, Dict[str, str]]:
        """Загрузка конфигурации моделей для мониторинга"""
        models = {}

        # Известные модели и их endpoints
        models['qwen2.5-vl-7b'] = {
            'endpoint': 'http://127.0.0.1:8077/v1',
            'model_name': 'Qwen2.5-VL-7B-Instruct',
            'engine': 'vLLM'
        }

        models['glm-4-9b-chat'] = {
            'endpoint': 'http://127.0.0.1:8077/v1',
            'model_name': 'glm-4-9b-chat',
            'engine': 'vLLM'
        }

        models['gemma-3-27b-awq'] = {
            'endpoint': 'http://localhost:8000/v2/models/gemma-3-27b-awq',
            'model_name': 'gemma-3-27b-awq',
            'engine': 'Triton vLLM',
            'custom_format': 'generate'  # Triton uses different format
        }

        models['deepseek-v4-flash'] = {
            'endpoint': 'http://31.28.10.135:8000/v1',
            'model_name': 'deepseek-v4-flash-dspark',
            'engine': 'vLLM (remote)',
            'status': 'remote'  # Remote model
        }

        return models

    def check_model_health(self, model_id: str) -> Dict[str, Any]:
        """Проверить здоровье одной модели"""
        if model_id not in self.models_config:
            return {
                'model_id': model_id,
                'status': 'unknown',
                'error': 'Model not in config'
            }

        config = self.models_config[model_id]
        endpoint = config['endpoint']
        model_name = config['model_name']

        health = {
            'model_id': model_id,
            'status': 'unknown',
            'endpoint': endpoint,
            'engine': config['engine'],
            'last_check': datetime.now().isoformat(),
            'response_time_ms': 0,
            'error': None
        }

        try:
            import requests
            import time

            # Quick health check
            start = time.time()

            if config.get('custom_format') == 'generate':
                # Triton format
                test_payload = {
                    "text_input": "Health check",
                    "stream": False
                }
                url = f"{endpoint}"
            else:
                # OpenAI format
                test_payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                }
                url = f"{endpoint.rstrip('/')}/chat/completions"

            r = requests.post(url, json=test_payload, timeout=10)
            health['response_time_ms'] = (time.time() - start) * 1000

            if r.status_code == 200:
                health['status'] = 'healthy'
            elif r.status_code == 404:
                health['status'] = 'model_not_found'
                health['error'] = f"404 - Model not found"
            elif r.status_code == 500:
                health['status'] = 'server_error'
                health['error'] = f"500 - Server error"
            else:
                health['status'] = f'http_{r.status_code}'
                health['error'] = f"HTTP {r.status_code}"

        except requests.exceptions.Timeout:
            health['status'] = 'timeout'
            health['error'] = 'Request timeout'
        except requests.exceptions.ConnectionError:
            health['status'] = 'unreachable'
            health['error'] = 'Cannot connect'
        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)[:100]

        return health

    def check_all_models(self) -> Dict[str, Any]:
        """Проверить здоровье всех моделей"""
        print("🔍 System Health Check (без прерывания процессов)")
        print("=" * 60)

        results = {}
        for model_id in self.models_config.keys():
            health = self.check_model_health(model_id)
            results[model_id] = health

            status_icon = "✅" if health['status'] == 'healthy' else "❌"
            print(f"{status_icon} {model_id:20} | {health['status']:15} | {health['response_time_ms']:.0f}ms")
            if health['error']:
                print(f"   Error: {health['error']}")

        # Summary
        healthy_count = sum(1 for r in results.values() if r['status'] == 'healthy')
        total_count = len(results)

        print(f"\n📊 Summary: {healthy_count}/{total_count} models healthy")

        return {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total': total_count,
                'healthy': healthy_count,
                'unhealthy': total_count - healthy_count
            }
        }

    def get_model_cards_status(self) -> Dict[str, Any]:
        """Проверить статус всех карточек"""
        print("\n📋 Model Cards Status Check")
        print("=" * 60)

        cards_info = []
        for card_file in self.base_dir.glob("out/card_*.json"):
            if ".bak." in card_file.name:
                continue

            try:
                with open(card_file) as f:
                    card = json.load(f)

                model = card.get('name', 'unknown')
                perf = card.get('perf', {})
                quality = card.get('quality', {})

                # Оценка завершенности
                completeness = 0.0
                if perf.get('ok', 0) > 0 and perf.get('n', 0) > 0:
                    completeness += 0.4  # Perf тесты успешны
                if quality.get('mmlu_pro', 0) > 0 or quality.get('humaneval', 'MISSING') != 'MISSING':
                    completeness += 0.3  # Есть quality метрики
                if quality.get('gsm8k', 0) > 0 or quality.get('minerva_math', 0) > 0:
                    completeness += 0.2  # Есть дополнительные метрики
                if card.get('micro', {}).get('math_n', 0) == 5:
                    completeness += 0.1  # Micro тесты

                cards_info.append({
                    'model': model,
                    'completeness': completeness,
                    'perf_ok': perf.get('ok', 0),
                    'perf_n': perf.get('n', 0),
                    'quality_metrics': len([k for k, v in quality.items() if v > 0]),
                    'has_errors': perf.get('ok', 0) < perf.get('n', 0)
                })

            except Exception as e:
                print(f"⚠️  Error reading {card_file.name}: {e}")

        # Сортировка по полноте
        cards_info.sort(key=lambda x: x['completeness'], reverse=True)

        for info in cards_info:
            status = "✅ ПОЛНЫЙ" if info['completeness'] >= 0.8 else "⚠️  ЧАСТИЧНЫЙ" if info['completeness'] >= 0.4 else "❌ ПУСТОЙ"
            print(f"{status} {info['model']:20} | {info['completeness']:5.1f} | perf: {info['perf_ok']}/{info['perf_n']} | quality: {info['quality_metrics']}/5")

        return {
            'timestamp': datetime.now().isoformat(),
            'cards': cards_info,
            'summary': {
                'total': len(cards_info),
                'complete': len([c for c in cards_info if c['completeness'] >= 0.8]),
                'partial': len([c for c in cards_info if 0.4 <= c['completeness'] < 0.8]),
                'empty': len([c for c in cards_info if c['completeness'] < 0.4])
            }
        }

    def get_active_processes(self) -> List[Dict[str, Any]]:
        """Проверить активные процессы"""
        print("\n🔄 Active Processes Check")
        print("=" * 60)

        try:
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=5
            )

            processes = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if not line.strip():
                    continue

                parts = line.split(None, 10)
                if len(parts) >= 11 and any(keyword in line.lower() for keyword in ['lm_eval', 'index', 'bench', 'qwen', 'gemma']):
                    processes.append({
                        'user': parts[0],
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'command': parts[10] if len(parts) > 10 else '',
                        'type': self._classify_process(line)
                    })

            for proc in processes:
                type_icon = "🧪" if proc['type'] == 'test' else "📊" if proc['type'] == 'index' else "🤖"
                print(f"{type_icon} PID: {proc['pid']:>8} | CPU: {proc['cpu']:>6}% | {proc['command'][:60]}")

            return processes

        except Exception as e:
            print(f"⚠️  Error checking processes: {e}")
            return []

    def _classify_process(self, process_line: str) -> str:
        """Классификация типа процесса"""
        line_lower = process_line.lower()
        if 'lm_eval' in line_lower:
            return 'test'
        elif 'index_batch' in line_lower:
            return 'index'
        elif 'bench' in line_lower or 'eval' in line_lower:
            return 'test'
        elif 'vllm' in line_lower or 'triton' in line_lower:
            return 'model'
        else:
            return 'unknown'

    def create_status_dashboard(self) -> str:
        """Создать HTML статус дашборд"""
        # Получаем данные
        health_results = self.check_all_models()
        cards_status = self.get_model_cards_status()
        processes = self.get_active_processes()

        # Создаем HTML
        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>System Status Dashboard</title>
    <style>
        body {{
            background: #0E1117;
            color: #E7EAF0;
            font-family: system-ui, sans-serif;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{ color: #F2B035; }}
        .status-card {{
            background: #161B24;
            border: 1px solid #29313F;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }}
        .status-icon {{ font-size: 24px; margin-right: 10px; }}
        .healthy {{ color: #57C97E; }}
        .unhealthy {{ color: #F2685C; }}
        .warning {{ color: #F2B035; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #29313F;
        }}
        th {{ color: #8B95A7; font-size: 12px; text-transform: uppercase; }}
        .good {{ color: #57C97E; }}
        .bad {{ color: #F2685C; }}
    </style>
</head>
<body>
    <h1>🔍 System Status Dashboard</h1>
    <p style="color: #8B95A7">Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="status-card">
        <div style="display: flex; align-items: center;">
            <span class="status-icon">{len([r for r in health_results['results'].values() if r['status'] == 'healthy'])}/{len(health_results['results'])}</span>
            <div>
                <h3>Model Health</h3>
                <p>Доступных моделей: {health_results['summary']['healthy']} из {health_results['summary']['total']}</p>
            </div>
        </div>
    </div>

    <div class="status-card">
        <div style="display: flex; align-items: center;">
            <span class="status-icon">{cards_status['summary']['complete']}</span>
            <div>
                <h3>Card Completeness</h3>
                <p>Полных: {cards_status['summary']['complete']}, Частичных: {cards_status['summary']['partial']}, Пустых: {cards_status['summary']['empty']}</p>
            </div>
        </div>
    </div>

    <h2>📊 Model Health Status</h2>
    <table>
        <thead><tr><th>Model</th><th>Status</th><th>Response Time</th><th>Engine</th><th>Error</th></tr></thead>
        <tbody>
"""

        for model_id, health in health_results['results'].items():
            status_class = "good" if health['status'] == 'healthy' else "bad"
            error_display = (health.get('error') or '')[:50]
            if 'NoneType' in str(error_display):
                error_display = 'None error'

            html += f"""
            <tr>
                <td>{model_id}</td>
                <td class="{status_class}">{health['status']}</td>
                <td>{health['response_time_ms']:.0f}ms</td>
                <td>{health['engine']}</td>
                <td>{error_display if error_display else '-'}</td>
            </tr>
"""

        html += """
        </tbody>
    </table>

    <h2>📋 Card Completeness</h2>
    <table>
        <thead><tr><th>Model</th><th>Completeness</th><th>Perf Status</th><th>Quality Metrics</th></tr></thead>
        <tbody>
"""

        for card in cards_status['cards']:
            status_class = "good" if card['completeness'] >= 0.8 else "bad" if card['completeness'] < 0.4 else "warning"

            html += f"""
            <tr>
                <td>{card['model']}</td>
                <td class="{status_class}">{card['completeness']:.1%}</td>
                <td>Perf: {card['perf_ok']}/{card['perf_n']}</td>
                <td>{card['quality_metrics']} metrics</td>
            </tr>
"""

        html += """
        </tbody>
    </table>

    <p style="color: #8B95A7; font-size: 12px;">
        💡 Для получения полных метрик запустите Phase 2 операции
    </p>
</body>
</html>"""

        return html

def main():
    monitor = SystemHealthMonitor()

    # Проверяем здоровье моделей
    monitor.check_all_models()

    # Проверяем статус карточек
    monitor.get_model_cards_status()

    # Проверяем активные процессы
    monitor.get_active_processes()

    # Создаем статус дашборд
    html = monitor.create_status_dashboard()

    # Сохраняем
    output_path = Path("/opt/llm-data/llmbench/out/system_status.html")
    output_path.write_text(html, encoding="utf-8")

    print(f"\n✅ Status dashboard создан: {output_path}")
    print(f"🌐 Открой в браузере: file://{output_path.absolute()}")

if __name__ == "__main__":
    main()
