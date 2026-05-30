#!/usr/bin/env python3
"""
CS599 Lab 7: 实验一 — CoT 思维链与闭环幻觉

验证 CoT 对多步推理的准确性提升，并亲手复现"闭环幻觉"——
当 CoT 第一步基于错误事实时，后续严密推理如何全盘崩溃。

运行:
    python experiment_A_cot.py
"""

import json
import os
import statistics
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL")

BASELINE_PROMPT = "请直接回答以下问题，只给出最终答案。\n问题: {question}\n答案:"

COT_PROMPT = (
    "请逐步思考以下问题，展示每一步推理过程，最后给出答案。\n"
    "问题: {question}\n让我们一步步思考："
)

POISONED_PROMPT = (
    "请基于以下已知事实进行推理。\n"
    "已知事实: {false_premise}\n\n"
    "请逐步思考以下问题，展示推理过程，最后给出答案。\n"
    "问题: {question}\n基于上述事实的推理："
)

QUESTIONS = [
    {
        "id": "Q1",
        "question": "一个水缸有3个水管，A管单独注满需4小时，B管需6小时，C管放空需8小时。"
                   "三管同时打开，多久注满水缸？",
        "false_premise": "水缸容量为500升，A管每小时注入100升，B管每小时注入80升，C管每小时放水62.5升。",
        "answer": "24/7 ≈ 3.43小时",
        "tolerance": 0.1,
    },
    {
        "id": "Q2",
        "question": "小明从家到学校，去程速度为5km/h，回程速度为3km/h，全程平均速度是多少？",
        "false_premise": "家到学校距离为30公里。",
        "answer": "3.75 km/h",
        "tolerance": 0.1,
    },
    {
        "id": "Q3",
        "question": "一批零件，甲单独做需10天，乙单独做需15天。两人合作2天后甲离开，"
                   "乙单独完成剩余工作，乙还需要多少天？",
        "false_premise": "甲每天做50个零件，乙每天做30个零件，总共有500个零件。",
        "answer": "10天",
        "tolerance": 0,
    },
    {
        "id": "Q4",
        "question": "一个两位数，十位数字与个位数字之和为12，交换位置后新数比原数大18。求原数。",
        "false_premise": "十位数字为8，个位数字为4。",
        "answer": "57",
        "tolerance": 0,
    },
    {
        "id": "Q5",
        "question": "一项工程，甲队独做20天完成，乙队独做30天完成。两队合作若干天后，"
                   "甲队调离，乙队又做5天完成。问甲乙合作了多少天？",
        "false_premise": "甲队每天做10个单位工作量，乙队每天做5个单位工作量，工程总量200个单位。",
        "answer": "10天",
        "tolerance": 0,
    },
]


def call_llm(prompt: str, client: OpenAI) -> str:
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def check_answer(response: str, expected: str, tolerance: float) -> bool:
    lowered = response.lower()
    if isinstance(expected, str):
        for keyword in expected.lower().split():
            if keyword.replace("≈", "").strip() in lowered:
                return True
    try:
        import re
        numbers = re.findall(r"[\d.]+", lowered)
        if numbers:
            val = float(numbers[-1])
            target = float(re.findall(r"[\d.]+", expected)[0])
            return abs(val - target) <= tolerance + 0.01
    except (ValueError, IndexError):
        pass
    return expected.lower().replace("≈", "").strip() in lowered


def run_comparison():
    client_kwargs = {}
    if BASE_URL:
        client_kwargs["base_url"] = BASE_URL
    client = OpenAI(**client_kwargs)

    print("=" * 60)
    print("  CS599 Lab 7: 实验一 — CoT 思维链与闭环幻觉")
    print("=" * 60)
    print(f"模型: {MODEL}")
    print(f"测试问题数: {len(QUESTIONS)}")
    print()

    results = {"baseline": [], "cot": [], "poisoned": []}

    for q in QUESTIONS:
        qid, question = q["id"], q["question"]
        false_premise = q["false_premise"]
        expected, tolerance = q["answer"], q["tolerance"]

        print(f"━━━ {qid} ━━━")
        print(f"问题: {question}")
        print(f"正确答案: {expected}")
        print()

        prompt_baseline = BASELINE_PROMPT.format(question=question)
        prompt_cot = COT_PROMPT.format(question=question)
        prompt_poisoned = POISONED_PROMPT.format(
            false_premise=false_premise, question=question,
        )

        print("  [Baseline 直接回答]")
        baseline_resp = call_llm(prompt_baseline, client)
        baseline_ok = check_answer(baseline_resp, expected, tolerance)
        print(f"    输出: {baseline_resp[:120]}{'...' if len(baseline_resp) > 120 else ''}")
        print(f"    结果: {'✅ 正确' if baseline_ok else '❌ 错误'}")
        results["baseline"].append(baseline_ok)

        print("  [CoT 逐步思考]")
        cot_resp = call_llm(prompt_cot, client)
        cot_ok = check_answer(cot_resp, expected, tolerance)
        print(f"    输出: {cot_resp[:120]}{'...' if len(cot_resp) > 120 else ''}")
        print(f"    结果: {'✅ 正确' if cot_ok else '❌ 错误'}")
        results["cot"].append(cot_ok)

        print("  [Poisoned CoT 错误前提]")
        print(f"    注入错误前提: {false_premise}")
        poisoned_resp = call_llm(prompt_poisoned, client)
        poisoned_ok = check_answer(poisoned_resp, expected, tolerance)
        print(f"    输出: {poisoned_resp[:120]}{'...' if len(poisoned_resp) > 120 else ''}")
        print(f"    结果: {'✅ 正确' if poisoned_ok else '❌ 错误（闭环幻觉！）'}")
        results["poisoned"].append(poisoned_ok)
        print()

    print("=" * 60)
    print("  综合对比")
    print("=" * 60)
    for mode, label in [("baseline", "Baseline (无CoT)"), ("cot", "CoT (逐步思考)"), ("poisoned", "Poisoned CoT (错误前提)")]:
        acc = sum(results[mode]) / len(results[mode]) * 100
        print(f"  {label:<30s}: {acc:.0f}% ({sum(results[mode])}/{len(results[mode])})")

    print()
    print("💡 核心发现：")
    print("  1. CoT 通过在上下文窗口中'外置工作记忆'，显著提升多步推理准确率。")
    print("  2. 但当第一步基于错误事实时，CoT 会将谬误传播至整个推理链——这就是闭环幻觉。")
    print("  3. Poisoned CoT 的输出往往仍然'逻辑严密'——这最危险，因为看不出错了。")
    print("  4. 打破闭环幻觉的关键：在推理中引入外部真实数据验证（→ 实验二 ReAct）。")


if __name__ == "__main__":
    run_comparison()
