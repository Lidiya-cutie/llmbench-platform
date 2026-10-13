#!/usr/bin/env python3
"""
Автоматическое сравнение моделей и генерация рекомендаций.
Создает интегральный отчет с лучшими моделями по категориям.

core-7 и Vision/OCR — раздельные домены (см. config/vision_domain.yaml).
"""

import json
from pathlib import Path
from datetime import datetime

CORE7 = ['mmlu_pro', 'gsm8k', 'minerva_math', 'humaneval', 'mbpp', 'ifeval', 'gpqa']
VISION = ['docvqa', 'ocrbench', 'chartqa', 'mmmu']


def load_cards():
    cards = []
    for card_file in Path('out').glob('card_*.json'):
        with open(card_file) as f:
            cards.append(json.load(f))
    return cards


def _score_from_card(card, bench):
    q = card.get('quality') or {}
    v = card.get('vision') or {}
    for src in (q, v):
        val = src.get(bench)
        if isinstance(val, (int, float)) and float(val) > 0:
            return float(val)
    return 0.0


def compare_by_benchmark(cards):
    results = {}
    for bench in CORE7:
        best_models = []
        for card in cards:
            score = card.get('quality', {}).get(bench, 0)
            if score and score > 0:
                best_models.append({
                    'name': card['name'],
                    'score': score,
                    'engine': card.get('engine', ''),
                    'quant': card.get('quant_format', ''),
                })
        if best_models:
            best_models.sort(key=lambda x: -x['score'])
            results[bench] = best_models[:3]
    return results


def compare_by_vision(cards):
    results = {}
    for bench in VISION:
        best = []
        for card in cards:
            mods = (card.get('modalities') or '').lower()
            if 'image' not in mods and 'vision' not in mods:
                continue
            score = _score_from_card(card, bench)
            if score > 0:
                best.append({
                    'name': card['name'],
                    'score': score,
                    'engine': card.get('engine', ''),
                    'status': (card.get('vision') or {}).get('status'),
                })
        if best:
            best.sort(key=lambda x: -x['score'])
            results[bench] = best[:3]
    return results


def calculate_overall_score(card):
    """Интегральная оценка core-7. Vision не входит."""
    quality = card.get('quality', {})
    weights = {
        'mmlu_pro': 1.2,
        'gsm8k': 1.0,
        'minerva_math': 1.0,
        'humaneval': 1.5,
        'mbpp': 1.2,
        'ifeval': 0.8,
        'gpqa': 1.1,
    }
    total_score = 0.0
    total_weight = 0.0
    for bench, weight in weights.items():
        score = quality.get(bench, 0) or 0
        if score > 0:
            total_score += score * weight
            total_weight += weight
    return total_score / total_weight if total_weight > 0 else 0


def get_active_benchmarks(card, benchmarks):
    active = []
    for bench in benchmarks:
        score = _score_from_card(card, bench)
        if score > 0:
            active.append(f'{bench}:{score:.3f}')
    return active


def get_use_case_recommendations(cards):
    use_cases = {
        'Чат-бот': {'mmlu_pro': 0.5, 'ifeval': 0.8, 'gsm8k': 0.6},
        'Программирование': {'humaneval': 1.5, 'mbpp': 1.2, 'mmlu_pro': 0.8},
        'Математика/Reasoning': {'gsm8k': 1.2, 'minerva_math': 1.0, 'mmlu_pro': 1.1, 'gpqa': 1.0},
        'RAG/Документы': {'mmlu_pro': 1.0, 'ifeval': 0.7},
        'Документы/OCR (Vision)': {'docvqa': 1.5, 'ocrbench': 1.2, 'chartqa': 0.8},
        'Инструкции': {'ifeval': 1.5, 'gsm8k': 0.5},
    }
    recommendations = {}
    for use_case, weights in use_cases.items():
        best_models = []
        for card in cards:
            use_case_score = 0.0
            total_weight = 0.0
            for bench, weight in weights.items():
                score = _score_from_card(card, bench)
                if score > 0:
                    use_case_score += score * weight
                    total_weight += weight
            if total_weight > 0:
                best_models.append({
                    'name': card['name'],
                    'score': use_case_score / total_weight,
                    'details': get_active_benchmarks(card, weights.keys()),
                })
        best_models.sort(key=lambda x: -x['score'])
        recommendations[use_case] = best_models[:3]
    return recommendations


def generate_comparison_report():
    cards = load_cards()
    print('=== АВТОМАТИЧЕСКОЕ СРАВНЕНИЕ МОДЕЛЕЙ ===')
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'Проанализировано моделей: {len(cards)}\n')

    print('ИНТЕГРАЛЬНАЯ ОЦЕНКА core-7 (с учетом весов):')
    print('   Веса: MMLU-Pro x1.2, GSM8K x1.0, Minerva-Math x1.0, HumanEval x1.5, MBPP x1.2, IFEval x0.8, GPQA x1.1')
    print('   Vision/OCR НЕ входит в core-7 (домен vision_ocr).\n')

    overall_scores = []
    for card in cards:
        score = calculate_overall_score(card)
        if score > 0:
            overall_scores.append({
                'name': card['name'],
                'score': score,
                'coverage': sum(
                    1
                    for k in CORE7
                    if (card.get('quality') or {}).get(k, 0) and (card.get('quality') or {}).get(k, 0) > 0
                ),
            })
    overall_scores.sort(key=lambda x: -x['score'])
    for i, model in enumerate(overall_scores[:5], 1):
        print(f"   {i}. {model['name']}: {model['score']:.3f} (покрытие: {model['coverage']}/7)")
    print()

    print('ЛУЧШИЕ РЕЗУЛЬТАТЫ ПО БЕНЧМАРКАМ core-7:\n')
    benchmark_results = compare_by_benchmark(cards)
    benchmark_names = {
        'mmlu_pro': 'MMLU-Pro (reasoning)',
        'gsm8k': 'GSM8K (math word problems)',
        'minerva_math': 'Minerva Math (math)',
        'humaneval': 'HumanEval (coding)',
        'mbpp': 'MBPP (coding)',
        'ifeval': 'IFEval (instructions)',
        'gpqa': 'GPQA (reasoning hard)',
    }
    for bench, best_models in benchmark_results.items():
        print(f'   {benchmark_names.get(bench, bench)}:')
        for i, model in enumerate(best_models, 1):
            print(f"      {i}. {model['name']}: {model['score']:.3f} ({model['engine']}, {model['quant']})")
        print()

    print('VISION / OCR (DocVQA, OCRBench) — отдельный домен:\n')
    vision_results = compare_by_vision(cards)
    vision_names = {
        'docvqa': 'DocVQA (documents)',
        'ocrbench': 'OCRBench (OCR)',
        'chartqa': 'ChartQA',
        'mmmu': 'MMMU',
    }
    if not vision_results:
        pending = []
        for card in cards:
            mods = (card.get('modalities') or '').lower()
            if 'image' in mods or 'vision' in mods:
                st = (card.get('vision') or {}).get('status', 'pending')
                pending.append(f"{card['name']} [{st}]")
        if pending:
            print('   Прогонов DocVQA/OCRBench пока нет. Кандидаты:')
            for p in pending:
                print(f'      - {p}')
            print('   Запуск: scripts/setup_vlmeval_venv.sh && LLMBENCH_VL_EXECUTE=1 scripts/run_vlm_docvqa_ocr.sh')
        else:
            print('   Нет vision-моделей в cards.')
        print()
    else:
        for bench, best_models in vision_results.items():
            print(f'   {vision_names.get(bench, bench)}:')
            for i, model in enumerate(best_models, 1):
                print(f"      {i}. {model['name']}: {model['score']:.3f} ({model.get('status')})")
            print()

    print('РЕКОМЕНДАЦИИ ПО СЦЕНАРИЯМ ИСПОЛЬЗОВАНИЯ:\n')
    recommendations = get_use_case_recommendations(cards)
    for use_case, best_models in recommendations.items():
        if best_models:
            print(f'   {use_case}:')
            for i, model in enumerate(best_models, 1):
                print(f"      {i}. {model['name']}: {model['score']:.3f}")
                if model['details']:
                    print(f"         Активные бенчмарки: {', '.join(model['details'][:3])}")
            print()

    print('ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:\n')
    balanced_models = [m for m in overall_scores if m['coverage'] >= 5][:3]
    if balanced_models:
        print('   Для универсального использования:')
        for model in balanced_models:
            print(f"      - {model['name']} — покрытие core-7 {model['coverage']}/7")
    print()
    coding = benchmark_results.get('humaneval', [])
    if coding:
        print('   Для генерации кода:')
        print(f"      - {coding[0]['name']} — HumanEval {coding[0]['score']:.3f}")
    print()

    report_data = {
        'generated_at': datetime.now().isoformat(),
        'overall_ranking': overall_scores,
        'benchmark_winners': benchmark_results,
        'vision_ocr_winners': vision_results,
        'use_case_recommendations': recommendations,
        'domains': {'core7': CORE7, 'vision_ocr': VISION},
    }
    with open('out/model_comparison_report.json', 'w') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print('Отчет сохранен: out/model_comparison_report.json')


if __name__ == '__main__':
    generate_comparison_report()
