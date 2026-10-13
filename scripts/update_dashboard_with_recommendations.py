#!/usr/bin/env python3
"""
Автоматическое обновление дашборда с рекомендациями.
Интегрирует сравнительный анализ в HTML дашборд.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def load_comparison_report():
    """Загружает отчет сравнения моделей"""
    try:
        with open('out/model_comparison_report.json') as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: Comparison report not found. Run auto_compare_models.py first")
        sys.exit(1)

def generate_recommendations_html(report):
    """Генерирует HTML с рекомендациями"""

    overall = report.get('overall_ranking', [])[:5]
    winners = report.get('benchmark_winners', {})
    use_cases = report.get('use_case_recommendations', {})

    html = f'''
<div class="recommendations-panel">
  <h3>Automatic Recommendations</h3>
  <p class="update-time">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

  <div class="recommendation-section">
    <h4>Top Models by Integral Score</h4>
'''

    for i, model in enumerate(overall[:3], 1):
        rank = "1st" if i == 1 else "2nd" if i == 2 else "3rd"
        border_color = "#4A90E2" if i == 1 else "#50E3C2" if i == 2 else "#F5A623"
        html += f'''
    <div class="top-model" style="border-left: 3px solid {border_color}">
      <span class="rank">{rank}</span>
      <span class="model-name">{model['name']}</span>
      <span class="score">{model['score']:.3f}</span>
      <span class="coverage">Coverage: {model['coverage']}/7</span>
    </div>
'''

    if overall:
        top_model = overall[0]
        html += f'''
    <p class="recommendation"><strong>{top_model['name']}</strong> - optimal choice with best integral score.</p>
'''

    html += '''
  </div>

  <div class="recommendation-section">
    <h4>Best Models by Use Case</h4>
'''

    scenario_icons = {
        'Чат-бот': 'chat',
        'Программирование': 'code',
        'Математика/Reasoning': 'math',
        'RAG/Документы': 'rag',
        'Инструкции': 'instructions'
    }

    for scenario, models in use_cases.items():
        if models:
            best = models[0]
            icon = scenario_icons.get(scenario, 'general')
            html += f'''
    <div class="scenario">
      <strong class="scenario-name">{scenario}:</strong>
      <span class="best">{best['name']} ({best['score']:.3f})</span>
'''

            if best.get('details'):
                html += f'      <p class="detail">Active benchmarks: {", ".join(best["details"][:3])}</p>'

            html += '    </div>\n'

    html += '''
  </div>

  <div class="recommendation-section">
    <h4>Benchmark Winners</h4>
'''

    benchmark_names = {
        'mmlu_pro': 'MMLU-Pro',
        'gsm8k': 'GSM8K',
        'minerva_math': 'Minerva Math',
        'humaneval': 'HumanEval',
        'mbpp': 'MBPP',
        'ifeval': 'IFEval',
        'gpqa': 'GPQA'
    }

    for bench, models in winners.items():
        if models:
            bench_name = benchmark_names.get(bench, bench)
            winner = models[0]
            html += f'''
    <div class="benchmark-winner">
      <span class="benchmark">{bench_name}:</span>
      <span class="winner">{winner['name']} - {winner['score']*100:.1f}%</span>
    </div>
'''

    html += '''
  </div>

  <div class="recommendation-section">
    <h4>Practical Recommendations</h4>
    <ul class="tips">
'''

    tips = []

    if overall:
        top = overall[0]
        tips.append(f"For general use: <strong>{top['name']}</strong> - best integral score ({top['score']:.3f})")

    if 'humaneval' in winners:
        coding_winner = winners['humaneval'][0]
        tips.append(f"For programming: <strong>{coding_winner['name']}</strong> - {coding_winner['score']*100:.1f}% on HumanEval")

    if 'gsm8k' in winners:
        math_winner = winners['gsm8k'][0]
        tips.append(f"For mathematics: <strong>{math_winner['name']}</strong> - {math_winner['score']*100:.1f}% on GSM8K")

    best_coverage = max(overall, key=lambda x: x['coverage'])
    if best_coverage['coverage'] >= 5:
        tips.append(f"For comprehensive testing: <strong>{best_coverage['name']}</strong> - maximum coverage ({best_coverage['coverage']}/7)")

    for tip in tips:
        html += f'      <li>{tip}</li>\n'

    html += '''
    </ul>
  </div>
</div>
'''
    return html

def update_dashboard_with_recommendations():
    """Обновляет дашборд с рекомендациями"""

    print("=== UPDATING DASHBOARD WITH RECOMMENDATIONS ===")

    report = load_comparison_report()
    print(f"OK: Loaded comparison report from {report.get('generated_at', 'unknown')}")

    recommendations_html = generate_recommendations_html(report)
    print("OK: Generated recommendations HTML")

    with open('dashboard_v2/recommendations_content.html', 'w', encoding='utf-8') as f:
        f.write(recommendations_html)

    print("OK: Saved file: dashboard_v2/recommendations_content.html")
    print("\nINTEGRATION INSTRUCTIONS:")
    print("  Add to index.html:")
    print("    <link rel='stylesheet' href='recommendations_content.html'>")

    full_report = {
        'updated_at': datetime.now().isoformat(),
        'comparison_report': report,
        'recommendations_html': recommendations_html,
        'top_model': report.get('overall_ranking', [{}])[0].get('name', 'unknown'),
        'best_score': report.get('overall_ranking', [{}])[0].get('score', 0)
    }

    with open('dashboard_v2/dashboard_with_recommendations.json', 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print("OK: Saved full report: dashboard_v2/dashboard_with_recommendations.json")
    print("\nDONE: Recommendations ready for dashboard integration")

if __name__ == '__main__':
    update_dashboard_with_recommendations()