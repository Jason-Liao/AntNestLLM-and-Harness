# -*- coding: utf-8 -*-
"""Agent 5：M1 直连演示 —— 本地 antNest LLM 驱动 Harness 完成真实任务。

模型-Harness 共同进化闭环的首次实跑：
  antNest LLM（SFT checkpoint）→ AntNestLLMClient → NestAgent Loop → 沙箱工具。
运行：PYTHONPATH=/workspace/antnest python -m antnest_harness.demo2
"""
from pathlib import Path

from .llm import AntNestLLMClient
from .tools import ToolRegistry
from .agent import NestAgent

REPORT = Path("/workspace/antnest/artifacts/harness_m1_report.md")


def main():
    tools = ToolRegistry()
    tools.register_defaults()
    client = AntNestLLMClient()
    print(f"已加载本地 antNest LLM: artifacts/{client.ckpt_name} "
          f"({sum(p.numel() for p in client.model.parameters())/1e3:.0f}K 参数)")

    agent = NestAgent(
        role="Agent Harness 团队",
        goal="由本地 antNest LLM 驱动，完成用户真实任务",
        backstory="模型与 Harness 共同进化的第一个执行者。",
        llm=client, tools=tools, max_iter=6)

    result = agent.kickoff("请检查 antNest 职位语料目录，并生成 M1 直连验证报告")
    print("\n── Agent Loop 轨迹 ──")
    for m in agent.memory.short_term:
        head = m["content"].replace("\n", " ")[:72]
        print(f"  [{m['role']:>9}] {head}")
    print("\n最终结果:", result)
    print("报告存在:", REPORT.exists())
    assert REPORT.exists(), "直连演示应产出报告"


if __name__ == "__main__":
    main()
