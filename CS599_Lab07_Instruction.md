# CS599 Lab 7：从直觉生成到逻辑推理——推理脚手架与智能体架构全景实验

> [!NOTE]
> **首席架构师寄语**：不要只调用 API，要理解推理的"结构"。本实验要求你徒手实现 CoT、ReAct、Plan-and-Solve、Tree of Thoughts 四种认知脚手架，深入 DeepSeek R1 的思维链内部探查"顿悟时刻"，并在工程实践中学会拦截循环陷阱、锚定推理目标。从 System 1 的直觉生成到 System 2 的逻辑推理——这趟旅程将重塑你对 LLM 智能边界的认知。

---

## 目录

- [1. 实验概述](#1-实验概述)
- [2. 环境准备](#2-环境准备)
- [3. 实验一：CoT 思维链——推理脚手架与闭环幻觉](#3-实验一cot-思维链推理脚手架与闭环幻觉)
- [4. 实验二：ReAct 从零实现——While 循环与 Stop Sequence](#4-实验二react-从零实现while-循环与-stop-sequence)
- [5. 实验三：Plan-and-Solve——规划解耦与长视距任务](#5-实验三plan-and-solve规划解耦与长视距任务)
- [6. 实验四：Tree of Thoughts——非线性全局搜索](#6-实验四tree-of-thoughts非线性全局搜索)
- [7. 实验五：原生 System 2 推理——DeepSeek R1 思维链探查](#7-实验五原生-system-2-推理deepseek-r1-思维链探查)
- [8. 实验六：工程挑战——循环陷阱拦截与目标锚定](#8-实验六工程挑战循环陷阱拦截与目标锚定)
- [9. 实验总结与综合思考](#9-实验总结与综合思考)
- [附录 A：术语表](#附录-a术语表)
- [附录 B：故障排查](#附录-b故障排查)

---

## 1. 实验概述

### 1.1 实验目标

本实验围绕 LLM 推理期计算的核心理念，以"数学问题求解与信息检索"为业务场景，完成以下六个相互关联的子实验：

| 实验 | 核心问题 | 推理模式 | 关键技术 |
|------|----------|----------|----------|
| 一 | CoT 如何提升推理？闭环幻觉如何产生？ | 线性推理链 | `Let's think step by step` + 错误前提传播 |
| 二 | 如何用外部工具打破闭环幻觉？ | 行动-观察闭环 | `while True` + `stop=["Observation:"]` + 工具调度 |
| 三 | 长视距任务如何防止"迷路"？ | 规划-执行解耦 | Planner Agent + Executor Agent |
| 四 | 遇到分叉决策，如何全局寻优？ | 树搜索 | BFS + 价值函数评分 + 剪枝回溯 |
| 五 | 推理能力能否内化为模型权重？ | 原生 System 2 | R1 `<think>` 推理链探查 + RLVR 机制 |
| 六 | 生产环境中如何防止循环陷阱与逻辑漂移？ | 工程干预 | 哈希拦截 + 高频目标锚定 + 事件驱动 |

### 1.2 前置知识

- 理解 System 1（直觉生成）与 System 2（逻辑推理）在 LLM 中的映射关系
- 掌握 LLM 推理 API 的基本调用方式（OpenAI 兼容接口）
- 具备 Python 基础，理解递归调用、正则解析、`asyncio` 异步编程
- 已安装并可用 `llama-server`（llama.cpp 推理服务器）

### 1.3 推理脚手架全景矩阵

本次实验覆盖四种推理脚手架模式，它们的机制、优劣势和成本差异如下：

| 模式 | 机制 | 优势 | 致命局限 | Token 成本 |
|------|------|------|----------|-----------|
| **CoT** | Transformer 自回归外部工作记忆 | 无需外部工具，提升数理准确率 | 闭环幻觉——中间出错全盘崩溃 | 低 |
| **Plan-and-Solve** | 先规划任务拓扑图再分步执行 | 消除长任务步骤遗漏 | 静态规划，环境变化适应性差 | 中 |
| **ReAct** | 行动-观察耦合的 While 循环状态转移 | 引入客观事实反馈，Grounding 最强 | 易陷入死循环，上下文急剧膨胀 | 较高 |
| **ToT** | 候选生成 + 价值评估 + BFS/DFS 回溯 | 全局寻优，解决极高难度决策 | 指数级 API 并发延迟，仅适用离线 | 极高 |

### 1.4 系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     Lab 7 实验脚本 (Python)                       │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ 实验一 (CoT)  │  │ 实验二 (ReAct)│  │ 实验三 (Plan&Solve)  │   │
│  │ 基础推理链     │  │ While循环控制 │  │ 规划器+执行器         │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────────┴───────────┐   │
│  │ 实验四 (ToT)  │  │ 实验五 (R1)  │  │ 实验六 (工程挑战)     │   │
│  │ BFS树搜索     │  │ 原生推理探查   │  │ 循环拦截+目标锚定    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ HTTP POST (OpenAI Compatible)
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  llama-server (:8080 / :8081)                                    │
│  Qwen2.5-7B / DeepSeek-R1-Distill-Qwen-7B                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐                   ┌──────────────────────┐
│  外部工具模拟器        │                   │  Phoenix 可观测性     │
│  - 计算器             │                   │  (:6006)             │
│  - 知识库检索         │                   │  OTLP Trace          │
│  - 天气/搜索 API 模拟 │                   │                      │
└──────────────────────┘                   └──────────────────────┘
```

---

## 2. 环境准备

### 2.1 Python 虚拟环境

```bash
python3 -m venv venv_cs599_lab7
source venv_cs599_lab7/bin/activate
pip install -r requirements.txt
```

### 2.2 LLM 推理服务

本实验需要 LLM 推理服务。**强烈建议使用本地 llama-server**，因为实验二需要精确控制 Stop Sequence，实验五需要使用 DeepSeek R1 Distill 模型。

#### 2.2.1 部署 llama-server

```bash
# 实验一/二/三/四/六（普通推理任务）
llama-server -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --port 8080 \
    --n-gpu-layers 99 \
    --ctx-size 8192 \
    --parallel 4

# 实验五（DeepSeek R1 推理，需要独立端口）
llama-server -m ~/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
    --port 8081 \
    --n-gpu-layers 99 \
    --ctx-size 8192 \
    --parallel 1
```

> **注意**：Qwen2.5-7B 和 DeepSeek R1 Distill 是两个不同的模型。前者用于实验一至四和六（通用推理），后者专门用于实验五（原生 System 2 思维链探查）。如 DeepSeek R1 模型未下载：
>
> ```bash
> huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF \
>     DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --local-dir ~/models/
> ```

#### 2.2.2 配置文件 `.env`

```bash
# 通用 LLM 配置（实验一/二/三/四/六）
OPENAI_API_KEY=sk-fake-key
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_MODEL=qwen2.5-7b

# DeepSeek R1 专用配置（实验五）
OPENAI_THINKING_BASE_URL=http://localhost:8081/v1
OPENAI_THINKING_MODEL=DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf

# Phoenix 可观测性（可选）
ENABLE_PHOENIX_TRACING=false
PHOENIX_COLLECTOR_ENDPOINT=http://127.0.0.1:6006/v1/traces
```

#### 2.2.3 使用 OpenAI API（备选方案）

如果选择使用云端 OpenAI 兼容 API（如 DeepSeek API）：

```bash
export OPENAI_API_KEY="sk-your-deepseek-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-chat"
```

> **注意**：使用云端 API 时，实验二的 Stop Sequence 行为取决于 API 服务端是否完整支持 `stop` 参数。llama.cpp 的实现最为原汁原味，推荐本地部署。

### 2.3 Phoenix 可观测性（可选）

```bash
docker run --rm -it \
  -p 6006:6006 \
  -p 4317:4317 \
  arizephoenix/phoenix:latest
```

在 `.env` 中设置 `ENABLE_PHOENIX_TRACING=true` 后，所有实验脚本会将 Trace 数据发送到 `http://localhost:6006`。

### 2.4 验证服务就绪

```bash
# 验证 Qwen2.5（端口 8080）
curl http://localhost:8080/health
# 应返回 {"status":"ok"}

# 验证 DeepSeek R1（端口 8081）
curl http://localhost:8081/health
```

### 2.5 实验项目结构

```
lab_7/
├── CS599_Lab07_Instruction.md   # 本实验指导书
├── README.md                     # 实验概览与快速开始
├── requirements.txt              # Python 依赖
├── .env                          # 环境变量配置
├── experiment_A_cot.py           # 实验一：CoT 思维链与闭环幻觉
├── experiment_B_react.py         # 实验二：ReAct 从零实现
├── experiment_C_plan_solve.py    # 实验三：Plan-and-Solve
├── experiment_D_tot.py           # 实验四：Tree of Thoughts
├── experiment_E_thinking.py      # 实验五：原生 System 2 思维链探查
├── experiment_F_engineering.py   # 实验六：工程挑战——循环陷阱与目标锚定
└── tools.py                      # 共享工具函数（模拟外部工具）
```

---

## 3. 实验一：CoT 思维链——推理脚手架与闭环幻觉

### 3.1 学习目标

- 理解 Chain of Thought (CoT) 如何利用 Transformer 自回归特性构建"外部工作记忆"
- 验证 CoT 对多步推理问题的准确性提升
- **亲手复现"闭环幻觉"（Closed-loop Hallucination）**：当 CoT 第一步基于错误前提时，后续严密推理如何全盘崩溃

### 3.2 理论解析

**CoT 的底层机制**：LLM 本质上是 System 1（直觉生成），通过自回归一次生成一个 Token。当让模型直接回答复杂问题时，跨度太大，逻辑容易断裂。CoT 的做法很简单——加上 `Let's think step by step`，让模型生成中间推理步骤。每一个中间步骤都通过自回归变成后续生成的上下文，相当于把外部工作记忆"挂载"到了 Context Window 里。跨步变小，逻辑就不会断。

**闭环幻觉的本质**：CoT 是封闭的"内心独白"——模型只与自己的上下文交互，不与外部真实世界接触。如果第一步基于概率生成了一个错误的事实基石（比如错误地假定了某个常数、算法或事实），后续再严密的逻辑推演都会建立在谬误之上。更致命的是，模型"无法证伪"——它没有机制去校验中间结论。

就像多米诺骨牌：第一块倒错了，后面全错。但模型输出看起来非常流畅、非常自信——这就是"逻辑自洽但事实荒谬"的深渊。

**本实验设计**：我们用三个对照实验来验证这一点：

| 组别 | Prompt 策略 | 预期行为 |
|------|------------|----------|
| **Baseline** | 直接提问，不加思考引导 | 准确率低，跨度太大会跳步 |
| **CoT** | 加入 `Let's think step by step` | 准确率大幅提升，逻辑清晰 |
| **Poisoned CoT** | CoT 但预先注入一个**错误的初始事实** | 模型基于错误前提推理，结论完全错误但自我感觉良好 |

### 3.3 代码分析

`experiment_A_cot.py` 的核心逻辑：

```python
# 三种提示策略
BASELINE_PROMPT = "请直接回答以下问题。\n问题: {question}\n答案:"

COT_PROMPT = (
    "请逐步思考以下问题，先分析再给出答案。\n"
    "问题: {question}\n"
    "让我们一步步思考："
)

POISONED_PROMPT = (
    "请基于以下已知事实思考。\n"
    "已知事实: {false_premise}  ← ⚠️ 这是错误前提！\n"
    "请逐步思考以下问题，先分析再给出答案。\n"
    "问题: {question}\n"
    "基于上述事实的推理："
)
```

测试问题设计为需要多步推理的数学/逻辑题，例如：
- "一个水缸有3个水管，A管注满需4小时，B管注满需6小时，C管放空需8小时。三管同时打开，多久注满？"
- 错误前提示例：把"水缸容量"或"流速"给一个错误值

### 3.4 执行步骤

```bash
python experiment_A_cot.py
```

### 3.5 预期结果

```
═══════════════════════════════════════════════════════
  CS599 Lab 7: 实验一 — CoT 思维链与闭环幻觉
═══════════════════════════════════════════════════════

任务: 水管注水问题...

【Baseline (无CoT)】
模型输出: 大约2.5小时 ← 可能正确也可能错误，无推理过程
准确率: 60% (3/5)

【CoT (逐步思考)】
模型输出: 设容量为V，A管流速V/4，B管流速V/6，C管流速-V/8
         净流速 = V/4 + V/6 - V/8 = 6V/24 + 4V/24 - 3V/24 = 7V/24
         时间 = V / (7V/24) = 24/7 ≈ 3.43小时
准确率: 100% (5/5) ✅

【Poisoned CoT (错误前提)】
已知事实: 水缸容量为500升，A管流速100升/小时，B管流速80升/小时 ← 错误数值！
模型推理: A流速100 L/h，B流速80 L/h，C放水62.5 L/h
         净流速 = 100 + 80 - 62.5 = 117.5 L/h
         时间 = 500 / 117.5 = 4.26小时 ← 完全错误！
         (模型输出非常自信但结论荒谬——这就是闭环幻觉)
准确率: 0% (0/5) ❌
```

### 3.6 思考题

1. 为什么 Poisoned CoT 的输出看起来仍然"逻辑严密"？这反映了 LLM 的什么本质特征？
2. 如果要在 CoT 中引入"自我验证"机制（在不使用外部工具的前提下），你会怎么设计 System Prompt？
3. CoT 的有效性与问题复杂度有什么关系？什么类型的问题 CoT 帮助最大？什么类型的问题 CoT 反而会"过度思考"（Overthinking）？

---

## 4. 实验二：ReAct 从零实现——While 循环与 Stop Sequence

### 4.1 学习目标

- 理解 ReAct（Reasoning + Acting）的 Thought → Action → Observation 循环
- **徒手实现 While 循环**：不借助任何 Agent 框架，用 `while True` 构建 ReAct 控制流
- **掌握 Stop Sequence**：理解 `stop=["Observation:"]` 为什么是 ReAct 的"刹车片"
- 验证 ReAct 如何通过外部工具反馈打破 CoT 的闭环幻觉

### 4.2 理论解析

**ReAct 的公式**：`Action_t = π(Observation_{t-1}, Thought_{t-1})`

智能体在 t 时刻的动作，由真实环境的观察和内部思考共同决定。这是一个人机（工具）协同的闭环：

```
大脑 (Thought) → 手部 (Action) → 外部环境 (Observation) → 回到大脑
```

**四步运转逻辑**：
1. **Thought**：System 2 规划——分析当前状态与目标，进行逻辑诊断
2. **Action**：System 1 生成——模型生成具体的工具调用指令（如 `Search[关键词]`）
3. **Observation**：**模型暂停生成**——外部系统执行指令，返回真实客观数据
4. **Repeat**：拼接客观事实，开启下一轮认知迭代

**Stop Sequence 的工程意义**：LLM 本质是文本补全机器——你不截断它，它会一直生成下去。如果不在生成 Action 后立即截断，模型会根据概率分布自己"脑补"出虚假的 Observation。这就是闭环幻觉的复辟！`stop=["Observation:"]` 强行截断模型生成，把控制权交还给 Python 运行时，让真实数据注入上下文。

**本实验设计**：我们模拟一个"智能助手搜索天气并计算"的场景。模型可以使用三个工具：
- `search_weather(city)` — 模拟搜索天气
- `search_population(city)` — 模拟搜索人口
- `calculate(expression)` — 数学计算

### 4.3 代码分析

`experiment_B_react.py` 的核心架构：

```python
# ══════════════════════════════════════════════
# ReAct 核心循环：徒手实现
# ══════════════════════════════════════════════

def react_loop(query: str, client: OpenAI, max_rounds: int = 10):
    history = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for round_num in range(max_rounds):
        # Step 1: 调用 LLM，stop=["Observation:"] 是关键刹车片！
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            stop=["Observation:"],  # ← 在模型将要"脑补"观察结果前截断
            temperature=0.0,
        )

        text = response.choices[0].message.content
        history.append({"role": "assistant", "content": text})

        # Step 2: 检查是否给出最终答案
        if "Final Answer:" in text:
            return extract_final_answer(text)

        # Step 3: 解析 Action（正则匹配 Action: tool[args]）
        action, args = parse_action(text)
        if not action:
            break  # 无法解析则终止

        # Step 4: 执行真实工具 → 获取真实观察结果
        observation = execute_tool(action, args)

        # Step 5: 将真实观察注入上下文
        history.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })

    return "达到最大循环次数，未能完成任务"
```

System Prompt 定义了 ReAct 格式：

```python
REACT_SYSTEM_PROMPT = """你是一个推理助手，使用以下格式解决任务：

Thought: [你的推理过程]
Action: tool_name[arguments]
Observation: [工具执行结果]
... (可以重复 N 次)
Thought: [基于观察结果的进一步推理]
Final Answer: [最终答案]

可用工具:
- search_weather[城市名] — 搜索城市天气
- search_population[城市名] — 搜索城市人口
- calculate[数学表达式] — 执行数学计算
"""
```

### 4.4 执行步骤

```bash
python experiment_B_react.py
```

交互式输入测试问题，例如：
- "上海的天气怎么样？比北京热吗？"
- "如果上海人口是北京的1.5倍，那么上海大约有多少人口？先查北京人口再计算。"

### 4.5 预期结果

```
═══════════════════════════════════════════════════
  CS599 Lab 7: 实验二 — ReAct 从零实现
═══════════════════════════════════════════════════

问题: 上海天气比北京热吗？

[Round 1]
Thought: 我需要查询上海和北京的天气。
Action: search_weather[上海]
─────────────────────────────────
[外部工具执行] search_weather("上海")
  返回真实数据: {"city": "上海", "temperature": 32, "condition": "晴"}

[历史记录] 追加 Observation: {"city": "上海", "temperature": 32, ...}

[Round 2]
Thought: 已获取上海32°C晴天，现在需要北京天气。
Action: search_weather[北京]
─────────────────────────────────
[外部工具执行] search_weather("北京")
  返回真实数据: {"city": "北京", "temperature": 28, "condition": "多云"}

[历史记录] 追加 Observation: {"city": "北京", "temperature": 28, ...}

[Round 3]
Thought: 上海32°C，北京28°C。上海比北京热4°C。
Final Answer: 是的，上海(32°C,晴)比北京(28°C,多云)温度高4°C。

✅ 完成！总轮次: 3

💡 Stop Sequence 的关键作用：
   - 每次生成 Thought + Action 后，stop=["Observation:"] 阻止了模型自行编造天气数据
   - 如果去掉 stop 参数，模型可能在 Action 后直接生成伪造的观察结果
   - 正是这个"刹车片"将控制权交还给 Python 运行时，执行真实工具
```

### 4.6 对比实验：验证 Stop Sequence 的必要性

脚本包含 `--no-stop` 开关，可以对比移除 `stop` 参数后的行为：

```bash
# 不使用 stop sequence（模型会自行"脑补" Observation）
python experiment_B_react.py --no-stop
```

预期看到模型在生成 Action 后**自行编造** Observation 内容（闭环幻觉复辟）。

### 4.7 思考题

1. 如果去掉 `stop=["Observation:"]`，模型为什么会在 Action 之后自己生成 Observation？这跟 LLM 的自回归本质有什么关系？
2. ReAct 循环中，每一次轮次都会将新的 Observation 追加到 history。如果任务需要 50 轮交互，Context Window 会受到什么影响？有什么解决策略？
3. 如果你要增加一个新工具（如 `translate[文本]`），需要修改哪些地方？这体现了 ReAct 模式怎样的扩展特性？

---

## 5. 实验三：Plan-and-Solve——规划解耦与长视距任务

### 5.1 学习目标

- 理解 Plan-and-Solve 中 Planner（规划者）与 Executor（执行者）的角色解耦
- 对比纯 CoT 与 Plan-and-Solve 在长视距任务上的表现
- 验证注意力衰减如何影响 CoT 但不影响 Plan-and-Solve

### 5.2 理论解析

**问题**：面对复杂长视距任务，CoT 的线性链条很容易"迷路"——随着思维链越来越长，模型对初始约束的注意力逐渐衰减，可能出现步骤遗漏或目标漂移。

**Plan-and-Solve 的解决方案**：解耦。

- **Planner Agent**：负责高维宏观规划。它不执行任何操作，只输出子任务的依赖拓扑图。Prompt 示例：`"Let's first understand the problem and devise a plan... Then carry out the plan step by step."`
- **Executor Agent**：拿着规划者拆解后的独立清单，在受限的上下文中逐一解决计算与语义问题。每个子任务的上下文很短，注意力不会衰减。

**核心优势**：切断规划与执行的物理耦合。Planner 不需要知道执行细节，Executor 不需要理解全局目标。每个 Agent 的上下文都很短，推理质量就很高。

**本实验设计**：设计一个"多步数据聚合计算"任务，例如：

> 查询以下5个城市（北京、上海、广州、深圳、杭州）的天气数据，计算平均温度，找出温度最高和最低的城市，然后按温度从高到低排序。

这个任务如果只用 CoT，思维链会非常长，模型容易在中间步骤"忘记"自己查到哪一步了。

### 5.3 代码分析

`experiment_C_plan_solve.py` 的核心架构：

```python
# ══════════════════════════════════════════════
# Phase 1: Planner — 生成子任务拓扑图
# ══════════════════════════════════════════════

PLANNER_PROMPT = """你是一个任务规划专家。请将以下复杂任务拆解为独立的子任务序列。
每个子任务必须是独立可执行的（不需要之前的上下文即可完成）。
只输出子任务列表，不要执行任何操作。

任务: {task}

格式:
1. [子任务描述]
2. [子任务描述]
...
"""

def generate_plan(task: str) -> list[str]:
    """Planner: 输出子任务列表"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PLANNER_PROMPT.format(task=task)}],
        temperature=0.0,
    )
    return parse_subtasks(response.choices[0].message.content)

# ══════════════════════════════════════════════
# Phase 2: Executor — 逐一执行子任务
# ══════════════════════════════════════════════

EXECUTOR_PROMPT = """请完成以下子任务。只给出结果，不要解释过程。
子任务: {subtask}
可用数据: {context}
结果:"""

def execute_subtask(subtask: str, results_so_far: dict) -> str:
    """Executor: 在受限上下文中执行单个子任务"""
    context = json.dumps(results_so_far, ensure_ascii=False)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": EXECUTOR_PROMPT.format(subtask=subtask, context=context),
        }],
        temperature=0.0,
    )
    return response.choices[0].message.content
```

### 5.4 CoT 对照组

脚本同时运行纯 CoT 模式作为对照：

```python
COT_COMPARISON_PROMPT = """请完成以下任务，逐步思考每一步。

任务: {task}

让我们一步步来："""
```

### 5.5 执行步骤

```bash
python experiment_C_plan_solve.py
```

### 5.6 预期结果

```
═══════════════════════════════════════════════════
  CS599 Lab 7: 实验三 — Plan-and-Solve 长视距任务
═══════════════════════════════════════════════════

任务: 查询5个城市天气，计算平均温度，找出极值，排序

━━━ Phase 1: Planner ━━━
规划者生成的子任务列表:
  1. 查询北京天气 → 记录温度
  2. 查询上海天气 → 记录温度
  3. 查询广州天气 → 记录温度
  4. 查询深圳天气 → 记录温度
  5. 查询杭州天气 → 记录温度
  6. 计算5个城市的平均温度
  7. 找出温度最高的城市
  8. 找出温度最低的城市
  9. 按温度从高到低排序

━━━ Phase 2: Executor ━━━
[1/9] 查询北京天气 → 北京: 28°C
[2/9] 查询上海天气 → 上海: 32°C
[3/9] 查询广州天气 → 广州: 35°C
[4/9] 查询深圳天气 → 深圳: 33°C
[5/9] 查询杭州天气 → 杭州: 30°C
[6/9] 计算平均温度 → (28+32+35+33+30)/5 = 31.6°C  ✅
[7/9] 最高温度 → 广州 35°C  ✅
[8/9] 最低温度 → 北京 28°C  ✅
[9/9] 排序 → 广州>深圳>上海>杭州>北京  ✅

━━━ CoT 对照组 ━━━
CoT 思维链:
第一步查了北京28°C，第二步查了上海32°C，第三步...
(中间开始混乱，查了3个城市后忘记了剩余2个城市)...
最终遗漏了杭州的数据  ❌ 步骤遗漏！

对比结论:
  Plan-and-Solve: 9/9 子任务完成，0 遗漏
  纯 CoT:         7/9 子任务完成，2 遗漏
  原因: CoT 的线性上下文导致注意力衰减，Planner 的拓扑图保证了完整性
```

### 5.7 思考题

1. 本实验中 Planner 和 Executor 使用同一个 LLM。如果使用不同成本的模型（如 Planner 用大模型、Executor 用小模型），会有什么影响？这在工程上有什么意义？
2. Plan-and-Solve 的 Planner 生成的是**静态计划**。如果执行到第 5 步时发现第 1 步的结果是错的，Plan-and-Solve 能否自动修正？如果不能，需要引入什么机制？
3. 对比 ReAct 和 Plan-and-Solve：什么时候用 ReAct？什么时候用 Plan-and-Solve？什么场景下两者可以组合？

---

## 6. 实验四：Tree of Thoughts——非线性全局搜索

### 6.1 学习目标

- 理解 Tree of Thoughts (ToT) 如何将推理升维为搜索问题
- 掌握 BFS（广度优先搜索）在文本推理中的实现
- 验证价值函数评分与剪枝机制的效果
- 量化对比 ToT 的精度提升与成本增加

### 6.2 理论解析

**ToT 的突破**：CoT 是线性的，ReAct 是循环的，但两者都有一个问题——一旦走错路就回不了头。ToT 把推理当作搜索问题，引入了计算机科学中经典的 BFS 和 DFS。

**ToT 三大核心机制**：

1. **候选生成**：在关键节点同时生成多个可能的 Thought 候选项（不只走一条路）
2. **价值函数**：利用 LLM 自身作为评估器，对每个分支评分——Sure（确定可行）、Likely（可能可行）、Impossible（不可能）
3. **主动回溯**：一旦识别死胡同，立刻剪枝并回溯探索其他分支

**代价**：ToT 需要大量并发 API 调用，Token 成本极高。这是"算力换智能"的典型。

**本实验设计**：使用一个经典的"24点游戏"（给定4个数，用加减乘除凑出24）作为测试任务。这个任务天然适合 ToT——每一步都有多个可能的操作分支。

### 6.3 代码分析

`experiment_D_tot.py` 的核心架构：

```python
# ══════════════════════════════════════════════
# ToT BFS 搜索
# ══════════════════════════════════════════════

def tot_bfs(
    problem: str,
    client: OpenAI,
    breadth: int = 3,      # 每层保留的分支数
    max_depth: int = 5,     # 最大搜索深度
) -> list[dict]:
    """
    Tree of Thoughts BFS 搜索
    """
    root = {"state": problem, "thoughts": [], "score": 1.0}
    frontier = [root]
    final_solutions = []

    for depth in range(max_depth):
        candidates = []

        # Step 1: 对 frontier 中每个节点生成候选 Thought
        for node in frontier:
            thoughts = generate_candidates(node["state"], client, breadth)
            for thought, new_state in thoughts:
                candidates.append({
                    "state": new_state,
                    "thoughts": node["thoughts"] + [thought],
                    "score": None,  # 待评估
                })

        # Step 2: 价值函数评估（LLM 评分）
        for candidate in candidates:
            score = evaluate_thought(candidate, client)
            candidate["score"] = score

            # 检查是否达到终止条件
            if score > 0.95:
                return candidate["thoughts"]

        # Step 3: 剪枝——只保留 Top-K 分支
        candidates.sort(key=lambda c: c["score"], reverse=True)
        frontier = candidates[:breadth]

        print(f"  深度 {depth + 1}: 生成 {len(candidates)} 个候选 → 保留 {len(frontier)} 个最佳分支")

    return None  # 未找到解

def generate_candidates(state: str, client: OpenAI, n: int) -> list[tuple[str, str]]:
    """为一个状态生成 n 个可能的下一步 Thought"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"当前问题状态: {state}\n\n"
                f"请生成 {n} 个可能的下一步推理方向。每个方向应显著不同。\n"
                f"格式: \n1. [方向一描述]\n2. [方向二描述]..."
            ),
        }],
        temperature=0.8,  # 提高创造性以生成多样化候选
        n=n,  # 一次调用生成 n 个候选
    )
    return parse_candidates(response)

def evaluate_thought(candidate: dict, client: OpenAI) -> float:
    """LLM 价值函数：评估当前推理路径的可行性"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"问题: {PROBLEM}\n"
                f"当前推理路径: {' → '.join(candidate['thoughts'])}\n"
                f"当前状态: {candidate['state']}\n\n"
                f"请评估此路径达到正确解的可能性。\n"
                f"输出: Sure / Likely / Impossible"
            ),
        }],
        temperature=0.0,
    )
    verdict = response.choices[0].message.content
    if "Sure" in verdict:
        return 0.95
    elif "Impossible" in verdict:
        return 0.05
    else:
        return 0.5  # Likely
```

### 6.4 执行步骤

```bash
python experiment_D_tot.py
```

### 6.5 预期结果

```
═══════════════════════════════════════════════════
  CS599 Lab 7: 实验四 — Tree of Thoughts BFS 搜索
═══════════════════════════════════════════════════

问题: 24点游戏 — 用数字 4, 7, 8, 8 通过加减乘除得到 24

━━━ ToT BFS 搜索过程 ━━━
根节点: 4, 7, 8, 8

  深度 1: 生成 8 个候选 → 保留 3 个最佳分支
    ✅ [Sure]    4 + 7 = 11, 剩余: 8, 8, 11
    ✅ [Sure]    8 - 7 = 1, 剩余: 4, 8, 1
    ⚠️ [Likely]  8 + 8 = 16, 剩余: 4, 7, 16
    ❌ [Impossible] 8 × 7 = 56 (太大，剪枝)

  深度 2: 生成 9 个候选 → 保留 3 个最佳分支
    ✅ [Sure]    11 + 8 = 19, 剩余: 8, 19
    ✅ [Sure]    16 + 4 = 20, 剩余: 7, 20
    ❌ [Impossible] 4 × 1 = 4 (回到类似状态，剪枝)

  深度 3: 生成 6 个候选 → 保留 3 个最佳分支
    ✅ [Sure]    19 + 8 = 27, 剩余: 27
    ❌ [Impossible] 20 - 7 = 13 (离24太远，剪枝)
    ...

  深度 4: 🎯 找到解！
    (8 - 7) = 1, 4 - 1 = 3, 8 × 3 = 24

━━━ 性能对比 ━━━

              找到解？  API 调用次数  Token 消耗
纯 CoT        否        1              ~200
ToT (BFS=3)   是        24             ~4800

💡 结论:
  - ToT 用 24 倍的成本换来了 CoT 无法找到的正确解
  - 剪枝机制在深度 2 处拦截了 3 条死胡同，阻止了大量无效的后续探索
  - "算力换智能"的典型体现：虽然贵，但能解决 CoT 无法解决的问题
```

### 6.6 思考题

1. 如果 BFS 的搜索宽度从 3 增加到 5，正确率会提升多少？Token 消耗会增加多少？这是一个什么类型的权衡？
2. 本实验用 LLM 自身作为价值函数。这种"自我评估"有没有局限性？如果模型本身的能力不足以判断某个分支是否可行，会发生什么？
3. ToT 在什么类型的任务上比 CoT 和 ReAct 有显著优势？（提示：考虑任务的"分叉点数量"和"单步可验证性"）

---

## 7. 实验五：原生 System 2 推理——DeepSeek R1 思维链探查

### 7.1 学习目标

- 理解 DeepSeek R1 基于 RLVR（带可验证奖励的强化学习）的训练机制
- 探查 R1 的 `<think>` 标签中隐藏的思维链
- 观察"顿悟时刻"（Aha Moment）——模型在推理中自发发现并纠正自己的错误
- 验证 `reasoning_content` 在多轮交互中的状态管理挑战

> [!IMPORTANT]
> **前置条件**：本实验需要 DeepSeek R1 Distill 模型。请确认已按 [2.2.1 节](#221-部署-llama-server) 在端口 8081 启动推理服务。

### 7.2 理论解析

**原生 System 2 模型**：前面四个实验的 CoT、ReAct、Plan-and-Solve、ToT 都是**外部的、Prompt 驱动的**认知脚手架。而 DeepSeek R1 和 OpenAI o1 这类模型代表了新范式——推理能力已经被训练到了模型权重里，不再需要外部脚手架。

**RLVR 的核心**：传统 RLHF 需要人类标注员团队加黑盒奖励模型。R1 的做法完全不同——对于数学和代码这类封闭域问题，利用编译器的客观规则直接赋予 +1 或 -1 的奖励。没有人类参与，没有主观判断。

**"Aha Moment"（顿悟时刻）**：在训练过程中，研究者观察到一个令人震撼的现象——模型在推理中突然意识到"等等，前面的推导存在逻辑漏洞，让我重新审视……"。机器自我演化出了严密的认知闭环。没有人类告诉它要"检查自己的答案"，它自己学会了。

**`<think>` 标签**：R1 在生成最终答案之前会生成一段内部推理过程，包裹在 `<think>...</think>` 标签中。这部分内容在 llama.cpp 的 OpenAI 兼容 API 中映射为 `reasoning_content` 字段。

### 7.3 代码分析

`experiment_E_thinking.py` 的核心探查逻辑：

```python
# ══════════════════════════════════════════════
# DeepSeek R1 思维链探查
# ══════════════════════════════════════════════

def probe_thinking(query: str, client: OpenAI) -> dict:
    """探查 R1 的完整思维链"""
    response = client.chat.completions.create(
        model=THINKING_MODEL,
        messages=[{"role": "user", "content": query}],
        temperature=0.6,
    )

    msg = response.choices[0].message
    msg_raw = msg.model_dump()

    # 提取原生 reasoning_content（llama.cpp 映射）
    reasoning = msg_raw.get("reasoning_content", "")
    # 同时从 content 中解析 <think> 标签
    think_content = extract_think_tags(msg.content or "")

    return {
        "reasoning_content": reasoning,
        "think_content": think_content,
        "final_content": msg.content or "",
        "finish_reason": response.choices[0].finish_reason,
    }
```

### 7.4 测试用例设计

本实验设计三类测试问题来探查 R1 的推理特征：

| 测试类型 | 示例问题 | 探查目标 |
|----------|---------|----------|
| **数学推理** | `计算 1+2+3+...+99+100 的和` | 观察 R1 如何分步推理、是否有公式推导 |
| **逻辑陷阱** | `树上有 5 只鸟，猎人开枪打死 1 只，还剩几只？` | 观察 R1 是否会先给出错误直觉再自我纠正（Aha Moment） |
| **代码调试** | `以下 Python 代码有什么 bug？def fib(n): return fib(n-1)+fib(n-2)` | 观察 R1 的分析深度和多角度检查 |

### 7.5 执行步骤

```bash
python experiment_E_thinking.py
```

### 7.6 预期结果

```
═══════════════════════════════════════════════════
  CS599 Lab 7: 实验五 — DeepSeek R1 原生思维链探查
═══════════════════════════════════════════════════

测试 1: 数学推理 — 1到100求和
─────────────────────────────────────────
[思维链探查: <think> 标签内容]
好的，用户问的是1加到100的和。
这其实是一个等差数列求和问题，公式是 n(n+1)/2。
让我验证：n=100，所以 100×(100+1)/2 = 100×101/2 = 5050。
但让我再想想，是不是也可以用高斯的方法...
(逐步推导，最终确认)
答案: 5050

[结论] R1 使用了公式推导，并进行了自我验证。

═══════════════════════════════════════════════════

测试 2: 逻辑陷阱 — 猎人打鸟问题
─────────────────────────────────────────
[思维链探查: <think> 标签内容]
树上有5只鸟，开枪打死1只...如果是常识问题，枪声会吓跑其他鸟，
所以剩下0只。但等等，我需要仔细考虑：
- 如果用的是消音猎枪呢？
- 死掉的那只鸟挂在树上呢？
...

等等，我意识到我刚才在过度思考。这是一个经典的思维陷阱题，
标准答案是枪声会吓跑所有鸟，所以剩下0只在树上。
但我应该给出推理过程，让用户知道我考虑过各种可能性...

[关键观察 🎯 Aha Moment!]
在推理中段，模型自发打断了之前的发散思考，回归到问题本质。

答案: 0只（枪声会吓跑其他鸟）

═══════════════════════════════════════════════════

测试 3: 代码调试 — 递归fib函数
─────────────────────────────────────────
[思维链探查: <think> 标签内容]
用户给的 fib 函数...让我分析一下：
def fib(n): return fib(n-1)+fib(n-2)

问题1: 没有基础情况（base case），会无限递归。
问题2: 没有处理 n=0 和 n=1。

正确的实现应该是：
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)

等等，我还注意到如果n很大，这个函数的效率非常低，
因为存在大量重复计算。时间复杂度是 O(2^n)。
可以优化为使用记忆化或迭代方式...

[结论] R1 不仅发现了主要 bug（缺少 base case），还自发分析了性能问题。
```

### 7.7 思考题

1. `<think>` 标签中的内容与 CoT 的"逐步思考"有什么本质区别？为什么说 R1 的推理是"内化的"而 CoT 是"外部的"？
2. 在逻辑陷阱题中观察到的"Aha Moment"——模型先发散再收敛——反映了推理的什么特征？这与 GRPO 算法中的"相对得分基线"有什么关系？
3. 如果需要在生产环境中使用 R1 的推理能力，`<think>` 标签内容是否应该暴露给最终用户？为什么？

---

## 8. 实验六：工程挑战——循环陷阱拦截与目标锚定

### 8.1 学习目标

- 复现 ReAct Agent 在生产环境中最常见的失败模式——无限循环
- 实现**哈希拦截**：检测连续重复调用并抛出硬中断
- 实现**高频目标锚定**：在循环中动态重新注入目标约束
- 理解**事件驱动架构**相对于盲目轮询的优势

### 8.2 理论解析

**三大生产环境挑战**：

1. **循环陷阱（Loop Trap）**：模型由于缺乏进展，在 Thought 与 Action 之间陷入无意义的死循环。每一轮循环都在烧 Token、烧钱。在没有人工值守的 7×24 生产环境中，一个死循环可能在一小时内烧掉数千美元。

2. **逻辑漂移（Logic Drift）**：随着思维链拉长，模型对初始约束的注意力衰减。一旦生成错误前提，它不仅不纠正，反而利用强大的语言能力编造严密逻辑来为自己辩护——这就是"元认知幻觉"。

3. **事件驱动 vs 轮询**：传统 while 循环是盲目的——即使外部工具在等待（如等待第三方 API 回调），Agent 仍然不断发起 LLM 调用"检查状态"。事件驱动架构让系统挂起等待外部信号，被唤醒后继续。

**本实验设计**：
- 设置一个"有陷阱的工具集"——某些工具参数如果设置不当会返回"请重试"或无效结果，诱导模型陷入循环
- 在基准版 ReAct 中观察死循环
- 在加固版中应用三种防御

### 8.3 代码分析

`experiment_F_engineering.py` 的核心防御逻辑：

```python
# ══════════════════════════════════════════════
# 防御 1: 哈希拦截 — 检测连续重复调用
# ══════════════════════════════════════════════

def harden_react_loop(query: str, client: OpenAI) -> dict:
    call_history = []  # 记录每次工具调用的哈希
    goal = query

    for round_num in range(max_rounds):
        # ══════════════════════════════════════
        # 防御 3: 高频目标锚定
        # ══════════════════════════════════════
        if round_num > 0 and round_num % 3 == 0:
            history.insert(0, {
                "role": "system",
                "content": f"[REMINDER] 你的最终目标是: {goal}。请勿偏离。",
            })

        response = client.chat.completions.create(
            model=MODEL, messages=history, stop=["Observation:"],
        )

        text = response.choices[0].message.content
        action, args = parse_action(text)

        # ══════════════════════════════════════
        # 防御 1: 哈希拦截
        # ══════════════════════════════════════
        call_hash = hashlib.md5(f"{action}{json.dumps(args)}".encode()).hexdigest()
        if call_history.count(call_hash) >= 2:
            raise LoopTrapException(
                f"检测到死循环！工具 {action} 已连续调用 3 次。"
                f"硬中断已触发，Agent 终止。"
            )
        call_history.append(call_hash)

        observation = execute_tool(action, args)
        # ...

# ══════════════════════════════════════════════
# 防御 2: 事件驱动模式（模拟）
# ══════════════════════════════════════════════

async def event_driven_agent(query: str):
    """事件驱动替代盲目轮询"""
    state = AgentState.IDLE

    # 提交异步任务
    task_id = await submit_async_task(query)
    state = AgentState.WAITING

    # 挂起等待（不烧 Token）
    while state == AgentState.WAITING:
        await asyncio.sleep(1)  # 模拟等待外部事件
        result = await check_task_status(task_id)
        if result:
            state = AgentState.PROCESSING
            break

    # 被唤醒后继续
    return await process_result(result)
```

### 8.4 执行步骤

```bash
# 基准版：无防御（观察死循环）
python experiment_F_engineering.py --baseline

# 加固版：哈希拦截 + 目标锚定
python experiment_F_engineering.py --defended
```

### 8.5 预期结果

```
═══════════════════════════════════════════════════
  CS599 Lab 7: 实验六 — 工程挑战与防御
═══════════════════════════════════════════════════

━━━ 基准版（无防御）━━━
任务: 查询不存在的城市"不夜城"的人口

[Round 1] Action: search_population[不夜城]
           Observation: {"error": "城市不存在"}
[Round 2] Action: search_population[不夜城]  ← 重复！
           Observation: {"error": "城市不存在"}
[Round 3] Action: search_population[不夜城]  ← 又重复！
...
[Round 10] 达到最大循环次数，Task Failed ❌

Token 消耗: 14,238 | 循环中无用调用: 9 次

━━━ 加固版（含防御）━━━
任务: 查询不存在的城市"不夜城"的人口

[Round 1] Action: search_population[不夜城]
           Observation: {"error": "城市不存在"}
[Round 2] Action: search_city_alias[不夜城]  ← 换了策略！
           Observation: {"suggestion": "是否指 '夜城'？"}
[Round 3] Action: search_population[夜城]
           Observation: {"city": "夜城", "population": 500000}
[Round 4] Final Answer: 不夜城可能指"夜城"，人口约50万 ✅

[🛡️ 防御日志]
  - 哈希拦截：未触发（模型成功切换策略）
  - 目标锚定：第3轮注入原始目标 ["查询不夜城人口"]

对比:
  基准版: 10轮死循环 → 失败，浪费14k tokens
  加固版: 4轮自适应 → 成功，消耗5k tokens
```

### 8.6 循环陷阱触发测试

脚本也包含一个主动触发循环陷阱的测试，使用一个"刁钻"的工具集：

```bash
python experiment_F_engineering.py --trap
```

预期看到：

```
[Round 1] Action: process_data[格式A]
           Observation: {"status": "retry", "message": "请重试"}

[Round 2] Action: process_data[格式A]  ← 模型重复尝试
           Observation: {"status": "retry", "message": "请重试"}

[Round 3] 🚨 哈希拦截触发！
           LoopTrapException: 检测到死循环！
           工具 process_data 已连续调用 3 次。
           硬中断已终止 Agent。

[对比] 无哈希拦截版本: Round 1 → 2 → 3 → ... → 10 (全部失败)
```

### 8.7 思考题

1. 哈希拦截虽然能检测到"完全相同"的重复调用，但如果模型每次微调参数（如 `process[格式A1]` `process[格式A2]`），它能检测到吗？如何改进？
2. 高频目标锚定在 LLM 的 context 最前端注入目标。如果 context 已经很长，模型还能有效"看到"这个锚定吗？这与 Transformer 的注意力机制有什么关系？
3. 在真实生产环境中，"事件驱动"和"轮询"各有什么优缺点？在什么场景下混合使用是最佳策略？

---

## 9. 实验总结与综合思考

### 9.1 核心结论回顾

通过六个实验，我们建立了从"直觉生成"到"逻辑推理"的完整认知链：

```
实验一（CoT）           → 推理脚手架基础：线性链条 + 闭环幻觉的致命性
实验二（ReAct）         → 外部工具闭环：While循环 + Stop Sequence + 打破幻觉
实验三（Plan-and-Solve） → 规划执行解耦：Planner + Executor 对抗注意力衰减
实验四（ToT）           → 非线性搜索：BFS + 价值函数 + 剪枝，算力换智能
实验五（R1 思维链）      → 原生推理内化：RLVR + <think>标签 + Aha Moment
实验六（工程挑战）        → 生产落地：哈希拦截 + 目标锚定 + 事件驱动
```

### 9.2 综合推理模型

将四种推理模式的差异整合为一个统一决策框架：

```
任务分析决策树：

1. 任务是否有可调用的外部工具？
   ├── 否 → 使用 CoT（简单任务）或 ToT（复杂分支决策）
   └── 是 → 进入 2

2. 任务是否超过 3 步？
   ├── 否 → 使用 ReAct
   └── 是 → 进入 3

3. 环境是否动态变化？
   ├── 是 → 使用 ReAct（需要实时观察反馈）
   └── 否 → 使用 Plan-and-Solve（静态规划更高效）

4. 任务是否有天然分叉点？
   ├── 是 → 对关键分叉节点使用 ToT
   └── 否 → 回到步骤 1

成本意识：
  CoT < Plan-and-Solve < ReAct < ToT
  （Token消耗从低到高）
```

### 9.3 开放性思考题

1. **架构组合**：如果你要设计一个"智能代码审查助手"，需要先读取代码结构（Plan-and-Solve），再逐函数深度分析（Executor），遇到可疑逻辑时需要搜索外部文档验证（ReAct），关键的架构决策需要多方案对比（ToT）。请设计这个 Agent 的架构图，标注各部分使用的推理模式。

2. **算力分配经济学**：实验四证明了 ToT 用 24 倍成本换来了 CoT 无法得到的答案。在什么场景下这个交易是值得的？什么场景下不值得？请从"任务价值"和"容错成本"两个维度分析。

3. **原生 vs 脚手架**：实验五展示了 R1 的原生推理能力。如果 R1 已经内化了推理能力，我们还需要实验一至四的外部脚手架吗？什么场景下原生推理不够，仍需要外部脚手架？

4. **工程防线**：实验六的三道防线（哈希拦截、目标锚定、事件驱动）中，哪一道是最基础、最不可或缺的？为什么？

---

## 附录 A：术语表

| 术语 | 全称 | 含义 |
|------|------|------|
| CoT | Chain of Thought | 思维链——通过逐步推理引导 LLM 解决复杂问题 |
| ReAct | Reasoning + Acting | 推理+行动——将 LLM 推理与外部工具调用融合 |
| ToT | Tree of Thoughts | 思维树——通过 BFS/DFS 在推理空间中搜索最优解 |
| BFS | Breadth-First Search | 广度优先搜索——逐层扩展候选分支 |
| DFS | Depth-First Search | 深度优先搜索——沿一条路走到黑再回溯 |
| RLVR | Reinforcement Learning with Verifiable Rewards | 基于可验证奖励的强化学习 |
| GRPO | Group Relative Policy Optimization | 分组相对策略优化——DeepSeek R1 的核心训练算法 |
| PPO | Proximal Policy Optimization | 近端策略优化——传统 RLHF 常用算法 |
| SFT | Supervised Fine-Tuning | 监督微调 |
| RLHF | Reinforcement Learning from Human Feedback | 基于人类反馈的强化学习 |
| System 1 / System 2 | — | 认知科学中的双系统理论：快直觉 vs 慢推理 |
| Aha Moment | 顿悟时刻 | R1 在推理中自发发现并纠正自己错误的时刻 |
| Stop Sequence | 停用词序列 | LLM 生成到特定文本时自动停止的机制 |
| Context Window Overflow (CWO) | 上下文窗口溢出 | 对话历史超出 LLM Token 上限 |
| HITL | Human-In-The-Loop | 关键操作需人工确认的安全机制 |
| Goal Anchoring | 目标锚定 | 在长循环中动态重新注入原始目标，对抗注意力衰减 |

---

## 附录 B：故障排查

### B.1 LLM 推理服务

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `Connection refused` | llama-server 未启动 | `curl http://localhost:8080/health` 确认 |
| Stop Sequence 不生效 | 模型或 API 服务端不支持 `stop` 参数 | 换用 llama.cpp 本地推理，确认参数通过 |
| DeepSeek R1 无 `<think>` 标签 | llama.cpp 版本太老 | 升级到最新版：`brew upgrade llama.cpp` |
| R1 `<think>` 内容被截断 | `--ctx-size` 太小 | 增大到 16384 或更大 |
| 实验二 ReAct 解析失败 | 模型未遵循 Prompt 格式 | 降低 temperature 到 0.0，检查 System Prompt |
| 实验五 temperature 过低 | R1 需要一定随机性才能展现"顿悟" | 建议 temperature 设为 0.6 |

### B.2 实验脚本

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `ModuleNotFoundError` | 未安装依赖 | `pip install -r requirements.txt` |
| ToT 实验耗时很长 | BFS 宽度/深度设置过大 | 减小 `--breadth` 和 `--max-depth` 参数 |
| Plan-and-Solve 计划不合理 | Planner 模型能力有限 | 可以尝试将 Planner 模型换成更强的大模型 API |
| 循环陷阱没有触发 | 模型的"智能"超出了陷阱设计 | 这是好现象，说明模型自主切换了策略 |

### B.3 DeepSeek R1 模型

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 模型文件不存在 | 未下载 R1 Distill | 按 2.2.1 节指令下载 |
| GPU 显存不足 (OOM) | 模型太大 | 确认使用 Q4_K_M 量化版本 (~4.7GB) |
| R1 回复乱码 | 量化精度太低 | 换用 Q5_K_M 或更高质量子版本 |

---

**版本**: v1.0
**最后更新**: 2026-05-29
