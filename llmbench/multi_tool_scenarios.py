"""
Многошаговые tool-calling сценарии для тестирования агентных возможностей.

Тесты с 2+ вызовами функций, цепочками и передачей контекста.
"""

import json
from typing import List, Dict, Any

# ------------------------- многошаговые сценарии -------------------------

MULTI_TOOL_SCENARIUS = [
    {
        "name": "weather_chain",
        "description": "Запрос погоды → сравнение температур → рекомендация",
        "messages": [
            {
                "role": "user",
                "content": "Какая погода сейчас в Лиссабоне и в Мадриде? "
                          "В каком из двух городов теплее и где лучше погода для прогулки?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Текущая погода в городе",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "Название города"}
                        },
                        "required": ["city"]
                    }
                }
            }
        ],
        "expected_calls": 2,
        "check_fn": lambda calls: (
            len(calls) >= 2 and
            any("get_weather" in str(c) and ("lisb" in str(c).lower() or "лисс" in str(c).lower()) for c in calls) and
            any("get_weather" in str(c) and ("madrid" in str(c).lower() or "мадрид" in str(c).lower()) for c in calls)
        )
    },
    {
        "name": "currency_conversion",
        "description": "Курс валют → конвертация → итоговая сумма",
        "messages": [
            {
                "role": "user",
                "content": "У меня 100 евро. Сколько я получу в долларах по текущему курсу? "
                          "И сколько это в рублях?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_exchange_rate",
                    "description": "Получить курс обмена между валютами",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string", "description": "Базовая валюта"},
                            "to": {"type": "string", "description": "Целевая валюта"}
                        },
                        "required": ["from", "to"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_currency",
                    "description": "Конвертировать сумму из одной валюты в другую",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {"type": "number", "description": "Сумма для конвертации"},
                            "from": {"type": "string", "description": "Исходная валюта"},
                            "to": {"type": "string", "description": "Целевая валюта"},
                            "rate": {"type": "number", "description": "Курс обмена"}
                        },
                        "required": ["amount", "from", "to", "rate"]
                    }
                }
            }
        ],
        "expected_calls": 2,
        "check_fn": lambda calls: (
            len(calls) >= 2 and
            any("get_exchange_rate" in str(c) for c in calls) and
            any("convert_currency" in str(c) and "100" in str(c) for c in calls)
        )
    },
    {
        "name": "search_and_calc",
        "description": "Поиск информации → вычисление → результат",
        "messages": [
            {
                "role": "user",
                "content": "Найди расстояние от Земли до Солнца в километрах и рассчитай, "
                          "сколько времени потребуется свету, чтобы преодолеть это расстояние "
                          "при скорости света. Ответ дай в часах."
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_fact",
                    "description": "Найти научный факт",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Запрос для поиска"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_time",
                    "description": "Рассчитать время пути при скорости света",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "distance_km": {"type": "number", "description": "Расстояние в км"},
                            "speed_kmh": {"type": "number", "description": "Скорость в км/ч"}
                        },
                        "required": ["distance_km", "speed_kmh"]
                    }
                }
            }
        ],
        "expected_calls": 2,
        "check_fn": lambda calls: (
            len(calls) >= 2 and
            any("search_fact" in str(c) and ("земля" in str(c).lower() or "солнц" in str(c).lower()) for c in calls) and
            any("calculate_time" in str(c) for c in calls)
        )
    },
    {
        "name": "booking_chain",
        "description": "Поиск отеля → проверка доступности → бронирование",
        "messages": [
            {
                "role": "user",
                "content": "Найди отели в центре Лиссабона с ценой до 150 евро за ночь, "
                          "проверь доступность на следующие выходные и забронируй самый дешевый вариант."
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_hotels",
                    "description": "Поиск отелей по критериям",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "Город"},
                            "max_price": {"type": "number", "description": "Максимальная цена"},
                            "location": {"type": "string", "description": "Район или расположение"}
                        },
                        "required": ["city", "max_price", "location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_availability",
                    "description": "Проверить доступность отеля",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hotel_id": {"type": "string", "description": "ID отеля"},
                            "check_in": {"type": "string", "description": "Дата заезда"},
                            "check_out": {"type": "string", "description": "Дата выезда"}
                        },
                        "required": ["hotel_id", "check_in", "check_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "book_hotel",
                    "description": "Забронировать отель",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hotel_id": {"type": "string", "description": "ID отеля"},
                            "check_in": {"type": "string", "description": "Дата заезда"},
                            "check_out": {"type": "string", "description": "Дата выезда"}
                        },
                        "required": ["hotel_id", "check_in", "check_out"]
                    }
                }
            }
        ],
        "expected_calls": 3,
        "check_fn": lambda calls: (
            len(calls) >= 3 and
            any("search_hotels" in str(c) and "lisbon" in str(c).lower() for c in calls) and
            any("check_availability" in str(c) for c in calls) and
            any("book_hotel" in str(c) for c in calls)
        )
    }
]


def score_multi_tool_scenario(message: dict, scenario: dict) -> bool:
    """Проверяет успешность многошагового сценария."""
    try:
        calls = (message or {}).get("tool_calls") or []
        check_fn = scenario["check_fn"]
        return check_fn(calls)
    except Exception as e:
        print(f"Error scoring scenario {scenario.get('name')}: {e}")
        return False


def get_multi_tool_result(messages: List[dict], tools_spec: List[dict],
                          scenario_name: str) -> Dict[str, Any]:
    """Запускает многошаговый сценарий и возвращает результат."""
    scenario = next((s for s in MULTI_TOOL_SCENARIUS if s["name"] == scenario_name), None)
    if not scenario:
        return {"error": f"Scenario {scenario_name} not found"}

    try:
        # Здесь будет логика вызова модели через API
        # Для сейчас возвращаем структуру для тестирования
        return {
            "scenario": scenario_name,
            "expected_calls": scenario["expected_calls"],
            "description": scenario["description"],
            "status": "ready_for_implementation"
        }
    except Exception as e:
        return {"error": str(e), "scenario": scenario_name}