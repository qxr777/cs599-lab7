#!/usr/bin/env python3
"""
CS599 Lab 7: 实验二 — ReAct 从零实现：While 循环 + Stop Sequence

徒手实现 ReAct (Thought → Action → Observation) 循环。
不借助任何 Agent 框架，用 while True + stop=["Observation:"] 构建控制流。

运行:
    python experiment_B_react.py              # 交互模式（使用 stop sequence）
    python experiment_B_react.py --no-stop    # 对比：不使用 stop sequence
    python experiment_B_react.py --demo       # 自动演示模式
"""

import argparse
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

REACT_SYSTEM_PROMPT = """你是一个推理助手。使用以下格式解决任务：

Thought: [你的分析推理过程，一次只思考一个 Action]
Action: tool_name[arguments]
Observation: [工具执行结果]
... (Thought/Action/Observation 可重复多次)
Thought: [基于所有观察结果的分析]
Final Answer: [最终答案]

可用工具:
- search_weather[城市名] — 搜索城市天气状况
- search_population[城市名] — 搜索城市人口数据
- calculate[表达式] — 执行数学计算（如 calculate[3*4+5]）

重要规则:
1. 每次回复只能包含一个 Thought 和一个 Action，或一个 Final Answer。
   不要在一个回复中同时包含 Action 和 Final Answer。
2. 必须严格按照 Thought/Action 格式，不要省略。
3. 使用 Action 之前必须有 Thought 说明为什么选择这个 Action。
4. 收到 Observation 后，必须在新的回复中做进一步 Thought。
5. Final Answer 必须是自然语言，包含关键数据，用中文回答。"""

ACTION_REGEX = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.DOTALL)
FINAL_ANSWER_REGEX = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
THOUGHT_REGEX = re.compile(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", re.DOTALL)

MAX_ROUNDS = 12


def parse_action(text: str) -> tuple[str | None, str | None]:
    m = ACTION_REGEX.search(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def extract_thoughts(text: str) -> list[str]:
    return [t.strip() for t in THOUGHT_REGEX.findall(text)]


def extract_final_answer(text: str) -> str | None:
    m = FINAL_ANSWER_REGEX.search(text)
    if m:
        return m.group(1).strip()
    return None


def execute_tool(action: str, args: str) -> str:
    executor = TOOL_EXECUTORS.get(action)
    if not executor:
        return json.dumps({"error": f"未知工具: {action}"}, ensure_ascii=False)
    try:
        result = executor(args)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def react_loop(query: str, client: OpenAI, use_stop: bool = True, verbose: bool = True) -> dict:
    history = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"请帮我解决以下问题：\n{query}"},
    ]

    stats = {
        "rounds": 0, "tool_calls": 0, "success": False,
        "answer": "", "thoughts": [], "tokens": 0,
    }

    for round_num in range(1, MAX_ROUNDS + 1):
        call_kwargs: dict = {
            "model": MODEL,
            "messages": history,
            "temperature": 0.0,
        }
        if use_stop:
            call_kwargs["stop"] = ["Observation:"]

        response = client.chat.completions.create(**call_kwargs)
        msg = response.choices[0].message
        text = msg.content or ""

        if verbose:
            print(f"\n{'─' * 40}")
            print(f"[Round {round_num}]")
            thoughts = extract_thoughts(text)
            for t in thoughts:
                print(f"  💭 {t[:200]}")
            action, action_args = parse_action(text)
            if action:
                print(f"  🔧 Action: {action}[{action_args}]")
            else:
                print(f"  📝 {text[:200]}")

        stats["rounds"] = round_num

        final_answer = extract_final_answer(text)
        if final_answer:
            stats["success"] = True
            stats["answer"] = final_answer
            if verbose:
                print(f"\n  ✅ Final Answer: {final_answer}")
            break

        action, action_args = parse_action(text)
        if not action:
            if verbose:
                print("  ⚠️ 无法解析 Action，自动终止")
            break

        stats["tool_calls"] += 1

        if use_stop:
            history.append({"role": "assistant", "content": text})
        else:
            user_msg = f"{text}\nObservation: "
            history.append({"role": "assistant", "content": text})
            history.append({"role": "user", "content": user_msg})

        observation = execute_tool(action, action_args)

        if use_stop:
            history.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        if verbose:
            obs_preview = observation[:150] + ("..." if len(observation) > 150 else "")
            print(f"  📊 Observation: {obs_preview}")

    return stats


def demo_queries():
    return [
        "上海的天气怎么样？比北京热吗？给出具体温度对比。",
        "北京和上海的总人口加起来是多少？不需要计算人口密度。",
        "如果上海人口比北京多，计算 (上海人口 - 北京人口) / 北京人口 的百分比。",
        "杭州天气温度，加上成都天气温度，减去西安天气温度，结果是多少？",
    ]


def main():
    parser = argparse.ArgumentParser(description="ReAct from Scratch")
    parser.add_argument("--no-stop", action="store_true", help="不使用 stop sequence")
    parser.add_argument("--demo", action="store_true", help="自动演示模式")
    args = parser.parse_args()

    use_stop = not args.no_stop

    client_kwargs = {}
    if BASE_URL:
        client_kwargs["base_url"] = BASE_URL
    client = OpenAI(**client_kwargs)

    mode_label = "STOP SEQUENCE 启用" if use_stop else "⚠️  STOP SEQUENCE 禁用（模型会自行脑补 Observation）"

    print("=" * 60)
    print("  CS599 Lab 7: 实验二 — ReAct 从零实现")
    print("=" * 60)
    print(f"模型: {MODEL}")
    print(f"模式: {mode_label}")
    print(f"最大轮次: {MAX_ROUNDS}")
    if use_stop:
        print("💡 stop=['Observation:'] — 阻止模型自行编造观察结果")
    else:
        print("⚠️  无 stop 参数 — 模型可能在 Action 后自行编造虚假 Observation")
    print()

    if args.demo:
        for query in demo_queries():
            print(f"\n{'═' * 60}")
            print(f"任务: {query}")
            print(f"{'═' * 60}")
            stats = react_loop(query, client, use_stop=use_stop)
            print(f"\n📊 统计: {stats['rounds']} 轮, {stats['tool_calls']} 次工具调用, "
                  f"{'✅ 成功' if stats['success'] else '❌ 失败'}")
            time.sleep(1)
        return

    print("交互模式 — 输入数学/逻辑/搜索问题，观察 ReAct 循环")
    print("输入 'quit' 退出\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break

        stats = react_loop(user_input, client, use_stop=use_stop)
        print(f"\n📊 统计: {stats['rounds']} 轮, {stats['tool_calls']} 次工具调用, "
              f"{'✅ 成功' if stats['success'] else '❌ 失败'}")
        print()


if __name__ == "__main__":
    main()
