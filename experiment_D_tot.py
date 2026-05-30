#!/usr/bin/env python3
"""
CS599 Lab 7: 实验四 — Tree of Thoughts (ToT) 非线性全局搜索

实现 BFS 搜索进行多路径推理探索，包含候选生成、价值函数评分、剪枝回溯。
对比 ToT 与纯 CoT 在复杂分支决策任务上的表现。

运行:
    python experiment_D_tot.py
    python experiment_D_tot.py --breadth 5 --max-depth 6
"""

import argparse
import json
import os
import re
import statistics

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL")

GENERATE_CANDIDATES_PROMPT = """你正在解决一个数学/逻辑问题。

当前问题: {problem}
当前状态（已完成的步骤）:
{current_state}

请生成 {n} 个可能的**下一步操作**。每个操作应显著不同，探索不同的解题路径。

格式:
1. [操作描述] → 新状态: [操作后的状态描述]
2. [操作描述] → 新状态: [操作后的状态描述]
...

只输出操作列表。"""

EVALUATE_PROMPT = """你正在评估一个解题路径的质量。

原始问题: {problem}
当前路径:
{path}

当前状态: {state}

请评估此路径到达正确答案的可能性。只用以下三个词之一回答:
- Sure: 非常确定路径正确
- Likely: 可能正确但不完全确定
- Impossible: 路径明显有误或走进死胡同

评估:"""


def parse_candidates(text: str) -> list[tuple[str, str]]:
    results = []
    pattern = re.compile(r"\d+\.\s+(.+?)\s*→\s*新状态:\s*(.+)", re.DOTALL)
    for m in pattern.finditer(text):
        desc = m.group(1).strip()
        state = m.group(2).strip()
        results.append((desc, state))
    if not results:
        lines = text.strip().split("\n")
        for line in lines:
            cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
            if cleaned:
                results.append((cleaned, cleaned))
    return results


def generate_candidates(problem: str, current_state: str, n: int, client: OpenAI) -> list[tuple[str, str]]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": GENERATE_CANDIDATES_PROMPT.format(
                problem=problem, current_state=current_state, n=n,
            ),
        }],
        temperature=0.8,
    )
    text = resp.choices[0].message.content or ""
    return parse_candidates(text)


def evaluate_thought(problem: str, path: list[str], state: str, client: OpenAI) -> float:
    path_str = " → ".join(path) if path else "(空)"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": EVALUATE_PROMPT.format(
                problem=problem, path=path_str, state=state,
            ),
        }],
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    if "sure" in verdict:
        return 0.95
    elif "impossible" in verdict:
        return 0.05
    else:
        return 0.5


def tot_bfs(
    problem: str, client: OpenAI,
    breadth: int = 3, max_depth: int = 5, verbose: bool = True,
) -> dict:
    stats = {"candidates_generated": 0, "api_calls": 0, "success": False}

    root = {"state": "初始状态", "thoughts": [], "score": 1.0}
    frontier = [root]

    for depth in range(1, max_depth + 1):
        if verbose:
            print(f"\n  ── 深度 {depth} ──")

        candidates = []
        for node in frontier:
            thoughts = generate_candidates(problem, node["state"], breadth, client)
            stats["api_calls"] += 1
            for desc, new_state in thoughts:
                candidates.append({
                    "state": new_state,
                    "thoughts": node["thoughts"] + [desc],
                    "score": None,
                })
        stats["candidates_generated"] += len(candidates)

        for candidate in candidates:
            score = evaluate_thought(problem, candidate["thoughts"], candidate["state"], client)
            stats["api_calls"] += 1
            candidate["score"] = score

            if score >= 0.95:
                if verbose:
                    print(f"    🎯 高置信度路径找到！分数: {score:.2f}")
                    print(f"    路径: {' → '.join(candidate['thoughts'][:3])}")
                stats["success"] = True
                stats["solution"] = candidate
                stats["depth"] = depth
                return stats

        candidates.sort(key=lambda c: c["score"], reverse=True)
        frontier = candidates[:breadth]

        if verbose:
            for i, c in enumerate(frontier[:3]):
                verdict = (
                    "Sure ✅" if c["score"] >= 0.9
                    else "Likely ⚠️" if c["score"] >= 0.5
                    else "Impossible ❌"
                )
                print(f"    [{i+1}] {verdict} {c['thoughts'][-1][:60]}...")

        if all(c["score"] < 0.1 for c in frontier):
            if verbose:
                print(f"    所有分支均评分过低，提前终止。")
            break

    stats["best"] = (
        max(frontier, key=lambda c: c["score"])
        if frontier else None
    )
    return stats


def solve_with_cot(problem: str, client: OpenAI) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"请解决以下问题，逐步思考并给出最终答案。\n\n问题: {problem}",
        }],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


TEST_PROBLEMS = [
    {
        "id": "P1",
        "problem": "用数字 4, 7, 8, 8 通过加减乘除运算得到 24。每个数字必须恰好使用一次。",
        "difficulty": "hard",
    },
    {
        "id": "P2",
        "problem": (
            "A、B、C、D、E 五人站成一排拍照。已知：A 不在两端；B 在 C 的左边；"
            "D 和 E 不相邻；C 不在最右边。请问从左到右的站位顺序是什么？"
        ),
        "difficulty": "medium",
    },
    {
        "id": "P3",
        "problem": (
            "一个农场有鸡和兔子共 35 只，这些动物总共有 94 条腿。"
            "问鸡和兔子各有多少只？"
        ),
        "difficulty": "easy",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Tree of Thoughts BFS Search")
    parser.add_argument("--breadth", type=int, default=3, help="BFS 搜索宽度")
    parser.add_argument("--max-depth", type=int, default=5, help="最大搜索深度")
    parser.add_argument("--problem", type=int, default=0, help="仅运行指定问题 (1-3)")
    args = parser.parse_args()

    client_kwargs = {}
    if BASE_URL:
        client_kwargs["base_url"] = BASE_URL
    client = OpenAI(**client_kwargs)

    print("=" * 60)
    print("  CS599 Lab 7: 实验四 — Tree of Thoughts BFS 搜索")
    print("=" * 60)
    print(f"模型: {MODEL}")
    print(f"搜索宽度: {args.breadth}, 最大深度: {args.max_depth}")
    print()

    problems = TEST_PROBLEMS
    if args.problem > 0:
        problems = [TEST_PROBLEMS[args.problem - 1]]

    for p in problems:
        pid, problem = p["id"], p["problem"]
        print(f"\n{'═' * 60}")
        print(f"  {pid}: {problem}")
        print(f"{'═' * 60}")

        print("\n  [ToT BFS 搜索]")
        tot_stats = tot_bfs(problem, client, breadth=args.breadth, max_depth=args.max_depth)

        print(f"\n  [CoT 基准]")
        cot_result = solve_with_cot(problem, client)
        print(f"    CoT 输出: {cot_result[:150]}{'...' if len(cot_result) > 150 else ''}")
        print(f"    CoT API 调用: 1 次")

        print(f"\n  📊 对比")
        print(f"    ToT API 调用: {tot_stats['api_calls']} 次")
        print(f"    ToT 候选生成: {tot_stats['candidates_generated']} 个")
        print(f"    ToT 成功: {'✅ 是' if tot_stats['success'] else '❌ 否'}")

        if tot_stats.get("solution"):
            sol = tot_stats["solution"]
            print(f"    ToT 路径: {' → '.join(sol['thoughts'][:3])}")

        if tot_stats["success"]:
            cost_ratio = tot_stats["api_calls"] / 1
            print(f"    成本比 (ToT/CoT): {cost_ratio:.1f}x")

    print(f"\n{'=' * 60}")
    print("💡 核心发现：")
    print("  1. ToT 用多路径搜索换来了 CoT 线性推理无法达到的决策深度。")
    print("  2. 价值函数评分有效剪枝，避免指数级搜索空间爆炸。")
    print("  3. ToT 的成本与搜索宽度 × 深度成正比——'算力换智能'。")
    print("  4. 任务分叉点越多、单步可验证性越强，ToT 优势越明显。")
    print("  5. 对于简单的线性问题（如鸡兔同笼），CoT 足够，ToT 过度。")

    print(f"\n  📈 API 成本估算 (breadth={args.breadth}, depth={args.max_depth}):")
    print(f"     最坏情况: ~{args.breadth * args.max_depth * 2} 次 API 调用")


if __name__ == "__main__":
    main()
