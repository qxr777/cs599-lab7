#!/usr/bin/env python3
"""
CS599 Lab 7: 实验六 — 工程挑战：循环陷阱拦截与目标锚定

复现 ReAct Agent 在生产环境中的三大挑战：
  1. 循环陷阱（Loop Trap）—— 相同工具调用反复失败
  2. 逻辑漂移（Logic Drift）—— 长时间运行后偏离原始目标
  3. 盲目轮询（Polling Tax）—— 每轮都在烧 Token

实现三种工程防御：
  1. 哈希拦截 —— 检测连续重复调用，硬中断
  2. 高频目标锚定 —— 动态重新注入原始目标约束
  3. 事件驱动模式 —— 挂起等待替代盲目轮询

运行:
    python experiment_F_engineering.py --baseline
    python experiment_F_engineering.py --defended
    python experiment_F_engineering.py --trap
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_EXECUTORS

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL")

REACT_SYSTEM_PROMPT = """你是一个推理助手。使用以下格式：

Thought: [分析推理]
Action: tool_name[arguments]
Observation: [工具结果]
... (重复)
Final Answer: [最终答案]

可用工具:
- search_weather[城市名] — 查询天气
- search_population[城市名] — 查询人口
- search_city_alias[城市别名] — 搜索城市别名或近似名
- process_data[格式名] — 处理数据（格式A/格式B/格式C）
- calculate[表达式] — 数学计算

规则:
1. 每次只输出一个 Thought + Action，或一个 Final Answer
2. 如果 Action 返回错误，尝试换一个策略而非重复相同调用
3. Final Answer 用中文，包含关键数据"""

ACTION_REGEX = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.DOTALL)

MAX_ROUNDS = 12


class LoopTrapException(Exception):
    pass


def parse_action(text: str) -> tuple[str | None, str | None]:
    m = ACTION_REGEX.search(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def execute_tool(action: str, args: str) -> str:
    executor = TOOL_EXECUTORS.get(action)
    if not executor:
        return json.dumps({"error": f"未知工具: {action}"}, ensure_ascii=False)
    try:
        result = executor(args)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def run_baseline(query: str, client: OpenAI) -> dict:
    """基准版：无任何防御"""
    history = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"请帮我解决以下问题：\n{query}"},
    ]

    stats = {"rounds": 0, "tool_calls": 0, "success": False, "answer": "",
             "repeated_calls": 0, "tokens_approx": 0, "loops_detected": 0}

    call_seen: dict[str, int] = {}

    for round_num in range(1, MAX_ROUNDS + 1):
        response = client.chat.completions.create(
            model=MODEL, messages=history, stop=["Observation:"], temperature=0.0,
        )
        msg = response.choices[0].message
        text = msg.content or ""

        if "Final Answer:" in text:
            stats["success"] = True
            stats["answer"] = text.split("Final Answer:")[-1].strip()
            stats["rounds"] = round_num
            break

        action, args = parse_action(text)
        if not action:
            stats["rounds"] = round_num
            break

        stats["tool_calls"] += 1
        stats["rounds"] = round_num

        call_hash = hashlib.md5(f"{action}{args}".encode()).hexdigest()[:12]
        call_seen[call_hash] = call_seen.get(call_hash, 0) + 1
        if call_seen[call_hash] >= 3:
            stats["repeated_calls"] += 1
            stats["loops_detected"] += 1

        observation = execute_tool(action, args)

        history.append({"role": "assistant", "content": text})
        history.append({"role": "user", "content": f"Observation: {observation}"})

    stats["tokens_approx"] = sum(len(json.dumps(m, ensure_ascii=False)) // 4 for m in history)
    return stats


def run_defended(query: str, client: OpenAI) -> dict:
    """加固版：哈希拦截 + 高频目标锚定"""
    goal = query
    history = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"请帮我解决以下问题。\n\n你的最终目标: {goal}\n\n问题: {query}"},
    ]

    stats = {"rounds": 0, "tool_calls": 0, "success": False, "answer": "",
             "repeated_calls": 0, "tokens_approx": 0, "loops_detected": 0,
             "anchors_injected": 0, "hard_interrupts": 0}

    call_history: list[str] = []

    for round_num in range(1, MAX_ROUNDS + 1):
        if round_num > 1 and round_num % 4 == 0:
            anchor = {
                "role": "system",
                "content": f"[🎯 目标锚定 #{round_num // 4}] 你的最终目标是: {goal}。请不要偏离目标。"
                           f"当前已进行 {round_num - 1} 轮，请继续朝向目标推进。",
            }
            history.insert(0, anchor)
            stats["anchors_injected"] += 1
            print(f"    🎯 目标锚定注入（第 {round_num} 轮）")

        response = client.chat.completions.create(
            model=MODEL, messages=history, stop=["Observation:"], temperature=0.0,
        )
        msg = response.choices[0].message
        text = msg.content or ""

        if "Final Answer:" in text:
            stats["success"] = True
            stats["answer"] = text.split("Final Answer:")[-1].strip()
            stats["rounds"] = round_num
            break

        action, args = parse_action(text)
        if not action:
            stats["rounds"] = round_num
            break

        stats["tool_calls"] += 1
        stats["rounds"] = round_num

        call_hash = hashlib.md5(f"{action}{json.dumps(args)}".encode()).hexdigest()
        if call_history.count(call_hash) >= 2:
            stats["loops_detected"] += 1
            stats["hard_interrupts"] += 1
            print(f"    🚨 [哈希拦截] 检测到循环！Action '{action}' 已连续调用 3 次！")
            print(f"    🔴 硬中断触发 — Agent 终止")
            stats["answer"] = "Agent 被硬中断终止：检测到死循环，无法完成任务。"
            break
        call_history.append(call_hash)

        observation = execute_tool(action, args)

        history.append({"role": "assistant", "content": text})
        history.append({"role": "user", "content": f"Observation: {observation}"})

    stats["tokens_approx"] = sum(len(json.dumps(m, ensure_ascii=False)) // 4 for m in history)
    return stats


async def event_driven_demo(query: str):
    """事件驱动模式演示（模拟异步等待）"""
    print("\n━━━ 事件驱动模式演示 ━━━")
    print(f"任务: {query}")

    tasks = [
        {"id": "task-1", "name": f"异步查询天气: {query}", "delay": 2.0, "result": {"temperature": 28}},
        {"id": "task-2", "name": "异步计算统计", "delay": 1.5, "result": {"avg": 30.5}},
    ]

    print("  [Agent] 提交异步任务...")

    async def run_async_task(task):
        print(f"    → {task['name']} 已提交，Agent 进入休眠（不烧 Token）")
        await asyncio.sleep(task["delay"])
        print(f"    ← {task['name']} 完成，结果: {task['result']}")
        return task["result"]

    results = await asyncio.gather(*[run_async_task(t) for t in tasks])

    print(f"  [Agent] 所有异步任务完成，恢复执行")
    print(f"  [Agent] 汇总结果: {json.dumps(results, ensure_ascii=False)}")
    print(f"  [对比] 如果使用轮询模式，每 1 秒一次 LLM 调用，")
    print(f"         等待 {max(t['delay'] for t in tasks)} 秒将浪费 {int(max(t['delay'] for t in tasks))} 次无意义 API 调用。")
    print(f"  [对比] 事件驱动模式在等待期间零 API 调用。")


def compare_stats(baseline_stats: dict, defended_stats: dict):
    print(f"\n{'─' * 60}")
    print(f"  对比分析")
    print(f"{'─' * 60}")
    print(f"  {'指标':<20s} {'基准版':<15s} {'加固版':<15s}")
    print(f"  {'─' * 50}")
    metrics = [
        ("轮次", "rounds", ""),
        ("工具调用", "tool_calls", ""),
        ("检测到循环", "loops_detected", ""),
        ("成功", "success", "✅/❌"),
        ("Token(估算)", "tokens_approx", ""),
        ("目标锚定注入", "anchors_injected", "次"),
        ("硬中断", "hard_interrupts", "次"),
    ]
    for label, key, suffix in metrics:
        b_val = baseline_stats.get(key, "N/A")
        d_val = defended_stats.get(key, "N/A")
        if isinstance(b_val, bool):
            b_val = "✅" if b_val else "❌"
            d_val = "✅" if d_val else "❌"
        print(f"  {label:<20s} {str(b_val) + suffix:<15s} {str(d_val) + suffix:<15s}")

    b_tokens = baseline_stats.get("tokens_approx", 0)
    d_tokens = defended_stats.get("tokens_approx", 0)
    if b_tokens and d_tokens:
        saved = ((b_tokens - d_tokens) / b_tokens) * 100
        print(f"\n  Token 节省: {saved:.0f}% ({b_tokens} → {d_tokens})")


def main():
    parser = argparse.ArgumentParser(description="工程挑战与防御")
    parser.add_argument("--baseline", action="store_true", help="基准版（无防御）")
    parser.add_argument("--defended", action="store_true", help="加固版（哈希拦截+目标锚定）")
    parser.add_argument("--trap", action="store_true", help="主动触发循环陷阱测试")
    parser.add_argument("--event", action="store_true", help="事件驱动模式演示")
    parser.add_argument("--compare", action="store_true", help="对比基准版与加固版")
    args = parser.parse_args()

    if not any([args.baseline, args.defended, args.trap, args.event, args.compare]):
        args.compare = True

    client_kwargs = {}
    if BASE_URL:
        client_kwargs["base_url"] = BASE_URL
    client = OpenAI(**client_kwargs)

    print("=" * 60)
    print("  CS599 Lab 7: 实验六 — 工程挑战与防御")
    print("=" * 60)
    print(f"模型: {MODEL}")
    print(f"最大轮次: {MAX_ROUNDS}")
    print()

    trap_query = "查询'不夜城'的人口数据，如果查不到就换个方式查。"

    if args.trap or args.baseline:
        print(f"\n{'═' * 60}")
        print("  基准版: 无防御 ReAct")
        print(f"{'═' * 60}")
        print(f"任务: {trap_query}")
        print()
        baseline_stats = run_baseline(trap_query, client)
        print(f"\n  📊 基准版统计:")
        print(f"     轮次: {baseline_stats['rounds']}, 工具调用: {baseline_stats['tool_calls']}")
        print(f"     循环检测: {baseline_stats['loops_detected']} 次")
        print(f"     成功: {'✅' if baseline_stats['success'] else '❌'}")
        print(f"     Token(估算): {baseline_stats['tokens_approx']}")
        if baseline_stats["loops_detected"] > 0:
            print(f"     💸 循环浪费了至少 {baseline_stats['loops_detected'] * 3} 次无意义工具调用！")

    if args.trap or args.defended:
        print(f"\n{'═' * 60}")
        print("  加固版: 哈希拦截 + 目标锚定")
        print(f"{'═' * 60}")
        print(f"任务: {trap_query}")
        print()
        defended_stats = run_defended(trap_query, client)
        print(f"\n  📊 加固版统计:")
        print(f"     轮次: {defended_stats['rounds']}, 工具调用: {defended_stats['tool_calls']}")
        print(f"     目标锚定注入: {defended_stats['anchors_injected']} 次")
        print(f"     硬中断: {defended_stats['hard_interrupts']} 次")
        print(f"     循环检测: {defended_stats['loops_detected']} 次")
        print(f"     成功: {'✅' if defended_stats['success'] else '❌'}")

    if args.trap or args.compare:
        print(f"\n{'═' * 60}")
        print("  对比实验")
        print(f"{'═' * 60}")
        print(f"任务: {trap_query}")
        print()
        bs = run_baseline(trap_query, client)
        time.sleep(0.5)
        ds = run_defended(trap_query, client)
        compare_stats(bs, ds)

    if args.event:
        print(f"\n{'═' * 60}")
        print("  事件驱动模式演示")
        print(f"{'═' * 60}")
        asyncio.run(event_driven_demo("查询北京天气并计算统计"))

    print(f"\n{'=' * 60}")
    print("💡 核心发现：")
    print("  1. 哈希拦截是最基础但最有效的循环防御——相同调用超过2次即触发硬中断。")
    print("  2. 但哈希拦截对'微调参数的重复调用'无效（如 process[格式A1] → process[格式A2]）。")
    print("  3. 高频目标锚定通过定期重新注入原始目标，有效对抗注意力衰减和逻辑漂移。")
    print("  4. 事件驱动架构将 Agent 从'盲目轮询'中解放，Token 消耗可能减少 50-90%。")
    print("  5. 在生产环境中，这三道防线应同时部署——纵深防御。")
    print("  6. 对 o1/R1 等原生推理模型，目标锚定需谨慎——"
          "过度注入可能干扰其原生推理链。")


if __name__ == "__main__":
    main()
