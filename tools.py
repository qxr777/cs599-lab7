"""
共享工具函数——模拟 ReAct / Plan-and-Solve / ToT 使用的外部工具。

可用工具:
  - search_weather(city)  → 搜索城市天气
  - search_population(city) → 搜索城市人口
  - calculate(expression)  → 数学计算
  - search_city_alias(name) → 搜索城市别名/近似名
  - process_data(format)    → 模拟数据处理（用于循环陷阱实验）
"""

import math
import re

WEATHER_DB = {
    "北京": {"city": "北京", "temperature": 28, "condition": "多云", "humidity": "45%"},
    "上海": {"city": "上海", "temperature": 32, "condition": "晴", "humidity": "60%"},
    "广州": {"city": "广州", "temperature": 35, "condition": "雷阵雨", "humidity": "80%"},
    "深圳": {"city": "深圳", "temperature": 33, "condition": "晴", "humidity": "70%"},
    "杭州": {"city": "杭州", "temperature": 30, "condition": "阴", "humidity": "55%"},
    "成都": {"city": "成都", "temperature": 26, "condition": "小雨", "humidity": "75%"},
    "西安": {"city": "西安", "temperature": 29, "condition": "晴", "humidity": "40%"},
}

POPULATION_DB = {
    "北京": {"city": "北京", "population": 21880000, "year": 2024},
    "上海": {"city": "上海", "population": 24870000, "year": 2024},
    "广州": {"city": "广州", "population": 18820000, "year": 2024},
    "深圳": {"city": "深圳", "population": 17790000, "year": 2024},
    "杭州": {"city": "杭州", "population": 12520000, "year": 2024},
    "成都": {"city": "成都", "population": 21400000, "year": 2024},
    "西安": {"city": "西安", "population": 13160000, "year": 2024},
}

CITY_ALIASES = {
    "不夜城": {"suggestion": "是否指 '夜城'？", "actual_match": "夜城"},
    "魔都": {"suggestion": "通常指 '上海'", "actual_match": "上海"},
    "帝都": {"suggestion": "通常指 '北京'", "actual_match": "北京"},
    "夜城": {"city": "夜城", "population": 500000, "temperature": 22, "condition": "雾"},
}


def search_weather(city: str) -> dict:
    if city in WEATHER_DB:
        return WEATHER_DB[city]
    if city in CITY_ALIASES and "city" in CITY_ALIASES[city]:
        return {
            "city": city,
            "temperature": CITY_ALIASES[city].get("temperature", "未知"),
            "condition": CITY_ALIASES[city].get("condition", "未知"),
            "humidity": "未知",
        }
    return {"error": f"未找到城市 '{city}' 的天气数据"}


def search_population(city: str) -> dict:
    if city in POPULATION_DB:
        return POPULATION_DB[city]
    return {"error": f"未找到城市 '{city}' 的人口数据"}


def calculate(expression: str) -> dict | str:
    allowed = re.sub(r"[^0-9+\-*/().%\s]", "", expression)
    try:
        result = eval(allowed, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"计算失败: {str(e)}"}


def search_city_alias(name: str) -> dict:
    if name in CITY_ALIASES:
        return CITY_ALIASES[name]
    return {"suggestion": None, "message": f"未找到 '{name}' 的相关别名"}


def process_data(format_name: str) -> dict:
    if format_name == "格式A":
        return {"status": "retry", "message": "请重试"}
    elif format_name == "格式B":
        return {"status": "retry", "message": "请重试"}
    elif format_name == "格式C":
        return {"status": "ok", "data": {"id": 42, "name": "测试数据"}}
    return {"status": "error", "message": f"未知格式: {format_name}"}


TOOL_EXECUTORS = {
    "search_weather": search_weather,
    "search_population": search_population,
    "calculate": calculate,
    "search_city_alias": search_city_alias,
    "process_data": process_data,
}
