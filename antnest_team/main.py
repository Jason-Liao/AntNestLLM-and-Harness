# -*- coding: utf-8 -*-
"""
antNest 团队入口。

用法：
  # 结构校验（不调用 LLM）：验证 32 个 Agent 一一对应职位、任务流依赖合法
  python main.py --validate

  # 打印团队花名册与任务流
  python main.py --list

  # 正式运行（需要配置 LLM，见下）
  python main.py --run

LLM 配置（环境变量，任选其一）：
  export OPENAI_API_KEY=sk-...                      # 默认走 OpenAI
  export ANTNEST_LLM="deepseek/deepseek-chat"       # 或任意 litellm 模型串
  export ANTNEST_LLM_API_BASE=https://...           # 自定义 OpenAI 兼容端点
  export ANTNEST_LLM_API_KEY=...
"""
import argparse
import os
import sys

from crewai import Crew, Process

from antnest_agents import build_agents
from antnest_tasks import build_tasks, OUT_DIR

POSITIONS = [
    "服务端开发工程师（大模型研究中台/线上核心服务/数据仓库）",
    "深度学习研发工程师",
    "预训练数据工程师",
    "AI 搜索算法/架构工程师",
    "Agent Harness 团队",
    "Agent Infra 研发工程师",
    "前端/客户端开发工程师",
    "测试开发工程师",
    "AI 跨界技术人才",
    "超算集群研发工程师",
    "高性能算子/通信/编译器工程师",
    "大模型训练/推理框架工程师",
    "高性能分布式存储工程师",
    "AI 平台运维工程师",
    "IT 基础设施团队",
    "IDC 数据中心团队",
    "AI 产品经理",
    "Code Agent 数据工程师",
    "通用 Agent 数据产品经理（办公/生活/搜索）",
    "专业领域数据产品经理（小语种、医学法律等学科）",
    "AI 创作数据产品经理",
    "情感智能数据产品经理",
    "Frontier（持续学习/自进化/新范式）研究员",
    "预训练（数据/算法）研究员",
    "后训练（数据/算法）研究员",
    "多模态理解（数据/算法）研究员",
    "HR 团队",
    "AGI 核心业务管培生",
    "法务团队",
    "财务团队",
    "采购团队",
    "行政团队",
]


def validate(agents: dict, tasks: list) -> bool:
    ok = True

    # 1) 32 个 Agent，一一对应 32 个职位
    if len(agents) != 32 or set(agents) != set(range(1, 33)):
        print(f"[FAIL] Agent 数量/编号异常: {len(agents)}")
        ok = False
    for i, pos in enumerate(POSITIONS, start=1):
        role = agents[i].role
        tag = "OK " if pos.split("（")[0].split("/")[0].strip() in role else "?? "
        print(f"  Agent {i:>2} | {pos}  ->  {role} {tag if tag.strip()!='OK' else ''}")

    # 2) 每个 Agent 至少承担一个任务
    tasked = {tk.agent for tk in tasks}
    missing = [i for i in agents if agents[i] not in tasked]
    if missing:
        print(f"[FAIL] 未承担任务的 Agent: {[POSITIONS[i-1] for i in missing]}")
        ok = False
    else:
        print(f"[OK] 32 个 Agent 全部参与任务（共 {len(tasks)} 个任务）")

    # 3) 任务 context 引用合法且均位于自身之前（顺序流程要求）
    order = {id(tk): n for n, tk in enumerate(tasks)}
    for n, tk in enumerate(tasks):
        contexts = tk.context if isinstance(tk.context, list) else []
        for ctx in contexts:
            if id(ctx) not in order:
                print(f"[FAIL] 任务{n+1} 引用了未知上下文")
                ok = False
            elif order[id(ctx)] >= n:
                print(f"[FAIL] 任务{n+1} 引用了后续任务的输出（顺序执行不合法）")
                ok = False
    if ok:
        print("[OK] 任务依赖全部合法（上下文均来自先前任务）")

    # 4) 输出目录
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[OK] 产出目录: {OUT_DIR}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="antNest 团队（32 Agents）")
    ap.add_argument("--validate", action="store_true", help="结构校验（不调用 LLM）")
    ap.add_argument("--list", action="store_true", help="打印团队与任务流")
    ap.add_argument("--run", action="store_true", help="正式运行团队")
    args = ap.parse_args()

    agents = build_agents()
    tasks = build_tasks(agents)

    if args.list or args.validate:
        print("=" * 70)
        print("antNest 团队 · 32 个 Agent（对应 antNest team 32 个职位）")
        print("=" * 70)
        for n, tk in enumerate(tasks, 1):
            print(f"  任务 {n:>2} | {tk.agent.role}")

    if args.validate:
        print("-" * 70)
        good = validate(agents, tasks)
        print("校验结果:", "通过 ✔" if good else "未通过 ✘")
        sys.exit(0 if good else 1)

    if args.run:
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        print("\n" + "=" * 70)
        print("antNest 产品开发总体方案（最终产出）")
        print("=" * 70)
        print(result.raw)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
