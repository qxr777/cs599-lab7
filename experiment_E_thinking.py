#!/usr/bin/env python3
"""
CS599 Lab 7: 实验五 — 原生 System 2 推理：DeepSeek R1 思维链探查

探查 DeepSeek R1 Distill 模型的 <think> 内部推理链。
观察 RLVR 训练产生的"顿悟时刻"（Aha Moment）。

前置条件：
  端口 8081 上运行 DeepSeek R1 Distill 模型

运行:
    python experiment_E_thinking.py
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

THINKING_MODEL = os.getenv("OPENAI_THINKING_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
THINKING_BASE_URL = os.getenv("OPENAI_THINKING_BASE_URL", os.getenv("OPENAI_BASE_URL"))

THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
XML_TAG_RE = re.compile(r"<[^>]+>")

TEST_CASES = [
    {
        "id": "Math-1",
        "type": "数学推理",
        "query": "计算 1+2+3+...+99+100 的和。请给出完整的推导过程。",
        "probe_target": "观察 R1 如何使用公式和高斯方法，是否自我验证",
    },
    {
        "id": "Logic-1",
        "type": "逻辑陷阱",
        "query": (
            "树上有 5 只鸟，猎人用枪打死 1 只，但用的是消音枪。"
            "请问树上还剩几只鸟？请仔细考虑所有可能性。"
        ),
        "probe_target": "观察 R1 是否会先发散后收敛（Aha Moment）",
    },
    {
        "id": "Debug-1",
        "type": "代码调试",
        "query": (
            "以下 Python 函数有什么问题？\n\n"
            "```python\ndef fib(n):\n"
            "    return fib(n-1) + fib(n-2)\n```\n\n"
            "请找出所有 bug 并给出修复方案。"
        ),
        "probe_target": "观察 R1 的多角度分析能力（base case、性能、边界条件）",
    },
    {
        "id": "Math-2",
        "type": "数学推理",
        "query": (
            "一个整数，如果它加上 100 后是一个完全平方数，"
            "再加上 168 又是一个完全平方数。求这个整数。"
        ),
        "probe_target": "观察 R1 的复杂数学推导和自我纠错能力",
    },
]


def extract_think_tag(text: str) -> str | None:
    m = THINK_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def clean_xml_tags(text: str) -> str:
    return XML_TAG_RE.sub("", text)


def analyze_reasoning(reasoning: str, think_content: str | None) -> dict:
    analysis = {
        "has_think_tag": think_content is not None,
        "reasoning_length": len(reasoning),
        "think_length": len(think_content) if think_content else 0,
        "has_self_correction": False,
        "has_formula": False,
        "has_multi_path": False,
    }

    combined = reasoning + (think_content or "")
    lowered = combined.lower()

    correction_patterns = [
        r"等等", r"不对", r"让我重新", r"让我再",
        r"wait", r"hold on", r"let me re-",
        r"actually", r"更正", r"仔细想想",
        r"我意识到", r"我注意到了", r"换个角度",
    ]
    for pat in correction_patterns:
        if re.search(pat, combined, re.IGNORECASE):
            analysis["has_self_correction"] = True
            break

    formula_patterns = [r"=.*[+\-*/].*=", r"公式", r"求和", r"方程"]
    for pat in formula_patterns:
        if re.search(pat, combined):
            analysis["has_formula"] = True
            break

    multi_path_patterns = [
        r"方法[一二三1-3]", r"另一种", r"或者", r"alternative",
        r"方案", r"路径", r"考虑.*\n.*考虑",
    ]
    for pat in multi_path_patterns:
        if re.search(pat, combined):
            analysis["has_multi_path"] = True
            break

    return analysis


def probe_thinking(query: str, query_type: str, client: OpenAI) -> dict:
    resp = client.chat.completions.create(
        model=THINKING_MODEL,
        messages=[{"role": "user", "content": query}],
        temperature=0.6,
    )

    msg = resp.choices[0].message
    content = msg.content or ""

    msg_raw = msg.model_dump()
    reasoning = msg_raw.get("reasoning_content", "")

    think_content = extract_think_tag(content)
    if think_content:
        clean_reasoning = think_content
    else:
        clean_reasoning = reasoning

    analysis = analyze_reasoning(reasoning, think_content)
    final_content = clean_xml_tags(content) if think_content else content

    return {
        "type": query_type,
        "query": query,
        "reasoning_content": reasoning,
        "think_content": think_content,
        "final_content": final_content,
        "full_content": content,
        "analysis": analysis,
    }


def print_result(result: dict):
    analysis = result["analysis"]
    think = result["think_content"]
    reasoning = result["reasoning_content"]

    display_think = think or reasoning

    print(f"\n{'─' * 50}")
    print(f"[思维链探查: {'<think> 标签' if think else 'reasoning_content 字段'}]")

    if display_think:
        preview = display_think[:500]
        if len(display_think) > 500:
            preview += f"\n... (总长度 {len(display_think)} 字符，已截断前500字符)"
        print(preview)
    else:
        print("(未检测到思维链内容 — 模型可能不支持 Thinking Mode 或字段名不同)")
        print(f"(原始响应前200字符: {result['full_content'][:200]})")

    print(f"\n{'─' * 50}")
    print(f"[最终回答]")
    final = result["final_content"]
    if think and final:
        cleaned_final = clean_xml_tags(final)
        print(cleaned_final[:300] if cleaned_final else final[:300])
    else:
        print(final[:300])

    print(f"\n{'─' * 50}")
    print(f"[推理特征分析]")
    print(f"  <think> 标签: {'✅ 检测到' if analysis['has_think_tag'] else '❌ 未检测到'}")
    print(f"  思维链长度: {analysis['reasoning_length']} 字符")
    print(f"  自我纠正 (Aha): {'✅ 检测到' if analysis['has_self_correction'] else '❌ 未检测到'}")
    print(f"  公式推导: {'✅ 检测到' if analysis['has_formula'] else '⚠️ 未检测到'}")
    print(f"  多路径探索: {'✅ 检测到' if analysis['has_multi_path'] else '  (未检测到)'}")

    if analysis["has_self_correction"]:
        print(f"\n  🎯 Aha Moment！模型在推理过程中进行了自我修正。")
        print(f"     这是 GRPO + RLVR 训练的产物——模型学会了在没有人类指导的情况下")
        print(f"     自发检查自己的工作并纠正错误。")


def main():
    client_kwargs = {}
    if THINKING_BASE_URL:
        client_kwargs["base_url"] = THINKING_BASE_URL
    client = OpenAI(**client_kwargs)

    print("=" * 60)
    print("  CS599 Lab 7: 实验五 — DeepSeek R1 原生思维链探查")
    print("=" * 60)
    print(f"模型: {THINKING_MODEL}")
    if THINKING_BASE_URL:
        print(f"API: {THINKING_BASE_URL}")
    print(f"测试用例: {len(TEST_CASES)} 个")
    print()

    all_has_think = 0
    all_has_aha = 0

    for tc in TEST_CASES:
        tid, qtype, query, probe = tc["id"], tc["type"], tc["query"], tc["probe_target"]
        print(f"\n{'═' * 60}")
        print(f"  {tid} — {qtype}")
        print(f"{'═' * 60}")
        print(f"问题: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"探查目标: {probe}")

        try:
            result = probe_thinking(query, qtype, client)
            print_result(result)

            if result["analysis"]["has_think_tag"]:
                all_has_think += 1
            if result["analysis"]["has_self_correction"]:
                all_has_aha += 1
        except Exception as e:
            print(f"\n  ❌ 错误: {e}")
            print(f"  💡 提示: 确保 DeepSeek R1 Distill 模型在端口 {THINKING_BASE_URL or '8081'} 运行")
            print(f"     如果使用的是普通模型（非 R1），则不会生成 <think> 标签。")

    print(f"\n{'═' * 60}")
    print(f"  总体统计")
    print(f"{'═' * 60}")
    print(f"  <think> 标签检出率: {all_has_think}/{len(TEST_CASES)}")
    print(f"  Aha Moment 检出率: {all_has_aha}/{len(TEST_CASES)}")

    print(f"\n{'=' * 60}")
    print("💡 核心发现：")
    print("  1. R1 的 <think> 标签内包含完整的推理过程，与 CoT 的外部 Prompt 驱动不同，")
    print("     这是**内化到模型权重中**的推理能力。")
    print("  2. Aha Moment 是 RLVR 训练的自然产物——模型通过 GRPO 算法学会自我纠错。")
    print("  3. R1 的推理链是多方向的：它会探索多个路径，然后收敛到最合理的答案。")
    print("  4. 与 Prompt 驱动的 CoT 不同，对 R1 输入 'think step by step' 反而可能")
    print("     干扰其原生推理逻辑（这是 OpenAI 对 o1 的已知警告）。")
    print("  5. <think> 内容的长度与问题复杂度正相关——模型会自适应地投入'思考时间'。")

    if all_has_think == 0:
        print()
        print("⚠️  注意：未检测到 <think> 标签。可能原因：")
        print("   1. 未使用 DeepSeek R1 模型（普通模型没有原生思维链）")
        print("   2. llama.cpp 版本不支持 reasoning_content 字段（升级到最新版）")
        print("   3. API 服务端不支持 DeepSeek 的 reasoning_content（换用 DeepSeek 官方 API）")


if __name__ == "__main__":
    main()
