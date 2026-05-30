# Lab 7: 从直觉生成到逻辑推理——推理脚手架与智能体架构全景实验

## 实验概述

本实验采用 **"徒手构建"（Build from Scratch）** 模式，不依赖任何 Agent 框架，从零实现 LLM 推理的四种认知脚手架——**CoT、ReAct、Plan-and-Solve、Tree of Thoughts**，并深入探查 **DeepSeek R1 的原生 System 2 推理机制**。

## 学习目标

- 徒手用 `while True` + `stop=["Observation:"]` 实现 ReAct 循环控制流
- 理解并复现 CoT 的"闭环幻觉"（Closed-loop Hallucination）
- 对比 Planner + Executor 解耦与纯线性推理在长视距任务上的差异
- 掌握 BFS 树搜索 + 价值函数评分在推理中的实现
- 探查 DeepSeek R1 的 `<think>` 内部推理链与"Aha Moment"
- 学会哈希拦截、目标锚定、事件驱动等工程防御手段

## 环境配置

```bash
# 1. 创建虚拟环境
python3 -m venv venv_cs599_lab7
source venv_cs599_lab7/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env  # 或手动创建 .env
```

## 启动推理服务

```bash
# 实验一/二/三/四/六（通用推理）
llama-server -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --port 8080 --n-gpu-layers 99 --ctx-size 8192 --parallel 4

# 实验五（DeepSeek R1 推理）
llama-server -m ~/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
    --port 8081 --n-gpu-layers 99 --ctx-size 8192 --parallel 1
```

## 项目结构

```
lab_7/
├── CS599_Lab07_Instruction.md   # 完整实验指导书
├── README.md                     # 本文件
├── requirements.txt              # Python 依赖
├── tools.py                      # 共享工具函数（天气/人口搜索、计算器）
├── experiment_A_cot.py           # 实验一：CoT 思维链与闭环幻觉
├── experiment_B_react.py         # 实验二：ReAct 从零实现
├── experiment_C_plan_solve.py    # 实验三：Plan-and-Solve
├── experiment_D_tot.py           # 实验四：Tree of Thoughts
├── experiment_E_thinking.py      # 实验五：R1 思维链探查
└── experiment_F_engineering.py   # 实验六：循环陷阱与工程防御
```

---

## 实验一：CoT 思维链与闭环幻觉

```bash
python experiment_A_cot.py
```

**核心发现：** CoT 通过"外置工作记忆"提升多步推理准确率。但当初始前提错误时，整个推理链全盘崩溃——这就是闭环幻觉。

---

## 实验二：ReAct 从零实现

```bash
python experiment_B_react.py                 # 交互模式（使用 stop sequence）
python experiment_B_react.py --demo          # 自动演示
python experiment_B_react.py --no-stop       # 对比：不使用 stop sequence
```

**核心发现：** `while True` + `stop=["Observation:"]` 是 ReAct 的工程核心。Stop Sequence 是"刹车片"——阻止模型自行编造虚假观察结果。

---

## 实验三：Plan-and-Solve

```bash
python experiment_C_plan_solve.py
```

**核心发现：** Planner 拆解任务拓扑图，Executor 在受限上下文中逐一执行。长视距任务中步骤遗漏率远低于纯 CoT。

---

## 实验四：Tree of Thoughts

```bash
python experiment_D_tot.py                          # 默认搜索参数
python experiment_D_tot.py --breadth 5 --max-depth 6  # 自定义搜索参数
```

**核心发现：** BFS + 价值函数评分 + 剪枝，用"算力换智能"。对分叉决策任务效果显著，但 API 调用次数 = 搜索宽度 × 深度 × 2。

---

## 实验五：R1 思维链探查

```bash
python experiment_E_thinking.py
```

**核心发现：** R1 的 `<think>` 标签内是内化到模型权重中的推理能力。在训练中自发出现了 Aha Moment——模型学会自我纠错，无需人类指导。

> **前置条件：** 需要 DeepSeek R1 Distill 模型在端口 8081 运行。

---

## 实验六：循环陷阱与工程防御

```bash
python experiment_F_engineering.py --baseline      # 基准版（观察死循环）
python experiment_F_engineering.py --defended      # 加固版（哈希拦截+目标锚定）
python experiment_F_engineering.py --trap          # 触发循环陷阱对比
python experiment_F_engineering.py --event         # 事件驱动模式演示
python experiment_F_engineering.py --compare       # 基准版 vs 加固版对比
```

**核心发现：** 哈希拦截（连续重复检测）+ 高频目标锚定（定期重新注入目标）+ 事件驱动（挂起等待替代轮询）构成生产环境的纵深防御体系。

---

## 实验提交物

1. **完整代码** — 所有 `.py` 文件
2. **运行截图** — 每个实验的控制台输出截图，包含关键数据
3. **实验报告** — 分析各推理模式的优劣势、成本对比和适用场景

## 参考资料

- **ReAct 论文：** Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (2022) — https://arxiv.org/abs/2210.03629
- **ToT 论文：** Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023) — https://arxiv.org/abs/2305.10601
- **DeepSeek R1 论文：** "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)
- **GRPO 论文：** "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)
- **《思考，快与慢》** — Daniel Kahneman (2011)
