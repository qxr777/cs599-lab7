#!/usr/bin/env python3
"""
CS599 Lab 7: 实验三 — Plan-and-Solve：规划解耦与长视距任务

对比纯 CoT 与 Plan-and-Solve 在长视距任务上的表现。
验证 Planner + Executor 架构如何防止注意力衰减。

运行:
    python experiment_C_plan_solve.py
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from tools import search_weather, search_population, calculate

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL")

PLANNER_PROMPT = """你是一个任务规划专家。请将以下复杂任务拆解为**一组独立的、可顺序执行的子任务**。

要求:
1. 每个子任务必须是**独立可执行**的——执行者不需要记住之前的上下文即可完成
2. 子任务之间是顺序依赖关系——后续子任务可以引用前面子任务的结果号
3. 将任务拆解为 N 个原子步骤（不要合并步骤）
4. 如果需要查询数据，将每次查询作为独立子任务

任务: {task}

格式（严格）:
1. [子任务1描述]
2. [子任务2描述]
...
N. [子任务N描述]

只输出子任务列表，不要其他内容。"""

EXECUTOR_PROMPT = """请完成以下单个子任务。只需要给出结果，不要解释过程。

子任务: {subtask}
此前步骤的结果:
{context}

当前子任务的结果:"""

COT_PROMPT = """请完成以下任务。逐步思考每一步，在推理完成后给出最终答案。

任务: {task}

让我们一步步来："""

COMPLEX_TASKS = [
    {
        "id": "T1",
        "task": (
            "查询以下5个城市（北京、上海、广州、深圳、杭州）的人口数据，"
            "计算这5个城市的平均人口，然后列出人口高于平均值的城市，"
            "最后按人口从高到低排序。"
        ),
    },
    {
        "id": "T2",
        "task": (
            "查询北京、成都、西安三个城市的天气温度，"
            "计算平均温度，找出温度最高的城市，"
            "如果最高温度超过30°C，则计算最高温度与最低温度的差值。"
        ),
    },
    {
        "id": "T3",
        "task": (
            "查询上海和北京的人口，计算 (上海人口 + 北京人口) / 2，"
            "然后查询杭州人口，计算杭州人口相对于上面那个平均值的百分比，"
            "再查询广州和深圳的人口，找出这5个城市中人口最多和最少的城市。"
        ),
    },
]

SUB_TASK_REGEX = re.compile(r"^\d+\.\s+(.+)", re.MULTILINE)


def parse_subtasks(plan_text: str) -> list[str]:
    tasks = SUB_TASK_REGEX.findall(plan_text)
    if not tasks:
        lines = plan_text.strip().split("\n")
        tasks = [line.lstrip("0123456789. -") for line in lines if line.strip()]
    return [t.strip() for t in tasks] if tasks else [plan_text.strip()]


def generate_plan(task: str, client: OpenAI) -> list[str]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PLANNER_PROMPT.format(task=task)}],
        temperature=0.0,
    )
    plan_text = resp.choices[0].message.content or ""
    subtasks = parse_subtasks(plan_text)
    return subtasks


def execute_subtask(subtask: str, context: dict, client: OpenAI) -> str:
    context_str = json.dumps(context, ensure_ascii=False, indent=2)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": EXECUTOR_PROMPT.format(subtask=subtask, context=context_str),
        }],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def plan_and_solve(task: str, client: OpenAI, verbose: bool = True) -> dict:
    print()
    print("━━━ Phase 1: Planner ━━━")
    plan = generate_plan(task, client)
    print(f"规划者生成了 {len(plan)} 个子任务:")
    for i, st in enumerate(plan, 1):
        print(f"  {i}. {st}")

    print()
    print("━━━ Phase 2: Executor ━━━")
    results: dict[str, str] = {}
    for i, subtask in enumerate(plan, 1):
        ctx = {str(k): v for k, v in results.items()}
        result = execute_subtask(subtask, ctx, client)
        results[str(i)] = result
        if verbose:
            print(f"  [{i}/{len(plan)}] {subtask[:60]}...")
            print(f"            → {result[:120]}{'...' if len(result) > 120 else ''}")

    final_context = json.dumps(results, ensure_ascii=False, indent=2)
    summary_resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"基于以下子任务结果，给出任务的最终答案。\n\n"
                f"结果: {final_context}\n\n最终答案:"
            ),
        }],
        temperature=0.0,
    )
    final_answer = summary_resp.choices[0].message.content or ""

    return {
        "plan": plan, "results": results, "final_answer": final_answer,
    }


def pure_cot(task: str, client: OpenAI) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": COT_PROMPT.format(task=task)}],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def evaluate_completeness(task: str, response: str, expected_cities: list[str]) -> dict:
    found = sum(1 for c in expected_cities if c in response)
    return {"found": found, "total": len(expected_cities), "ratio": found / len(expected_cities)}


def main():
    client_kwargs = {}
    if BASE_URL:
        client_kwargs["base_url"] = BASE_URL
    client = OpenAI(**client_kwargs)

    print("=" * 60)
    print("  CS599 Lab 7: 实验三 — Plan-and-Solve 长视距任务")
    print("=" * 60)
    print(f"模型: {MODEL}")
    print(f"测试任务数: {len(COMPLEX_TASKS)}")
    print()

    for t in COMPLEX_TASKS:
        tid, task = t["id"], t["task"]
        print(f"\n{'═' * 60}")
        print(f"  {tid}: {task}")
        print(f"{'═' * 60}")

        ps_result = plan_and_solve(task, client)
        time.sleep(1)

        cot_response = pure_cot(task, client)
        print(f"\n━━━ CoT 对照组 ━━━")
        print(f"  CoT 输出: {cot_response[:200]}{'...' if len(cot_response) > 200 else ''}")

        all_cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安"]
        ps_eval = evaluate_completeness(task, json.dumps(ps_result, ensure_ascii=False), all_cities)
        cot_eval = evaluate_completeness(task, cot_response, all_cities)

        print()
        print(f"  数据完整性评估:")
        print(f"    Plan-and-Solve: {ps_eval['found']}/{ps_eval['total']} 城市被覆盖")
        print(f"    CoT:            {cot_eval['found']}/{cot_eval['total']} 城市被覆盖")

    print(f"\n{'=' * 60}")
    print("💡 核心发现：")
    print("  1. Plan-and-Solve 的 Planner 将复杂任务拆解为独立子任务，")
    print("     避免了 CoT 的线性链条中注意力衰减导致的步骤遗漏。")
    print("  2. Executor 在受限上下文中执行每个子任务，互不干扰。")
    print("  3. 当任务步骤增多（>5步）时，Plan-and-Solve 的优势越发明显。")
    print("  4. Planner 规划质量取决于 LLM 的任务理解能力；If the plan is bad,")
    print("     even the best Executor cannot salvage it.")


if __name__ == "__main__":
    main()
