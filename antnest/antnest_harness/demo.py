# -*- coding: utf-8 -*-
"""Agent 5：Harness 离线演示（MockLLM 驱动，真实工具执行）。

任务：统计 antNest team 32 个职位文件并产出报告。
运行：PYTHONPATH=/workspace/antnest python -m antnest_harness.demo
配置真实 LLM 后（ANTNEST_LLM_API_BASE/KEY/MODEL），同一代码路径直接可用。
"""
import os
from pathlib import Path

from .llm import MockLLM, OpenAICompatClient
from .tools import ToolRegistry
from .agent import NestAgent

REPORT = Path("/workspace/antnest/artifacts/harness_demo_report.md")


def make_llm():
    if os.environ.get("ANTNEST_LLM_API_BASE"):
        return OpenAICompatClient()
    return MockLLM(script=[  # 离线脚本：list → shell 统计 → 写报告 → finish
        '```action\n{"action":"tool","name":"list_dir","args":{"p":"/workspace/extracted"}}\n```',
        '```action\n{"action":"tool","name":"shell","args":{"cmd":"ls /workspace/extracted | wc -l"}}\n```',
        '```action\n{"action":"tool","name":"write_file","args":{"p":"%s","c":"# antNest Harness 演示报告\\n\\n由 NestAgent（MockLLM 驱动 + 真实沙箱工具）生成：已完成 antNest team 职位文件统计任务。"}}\n```' % REPORT,
        '```action\n{"action":"finish","result":"已统计职位文件并写入 harness_demo_report.md"}\n```',
    ])


def main():
    tools = ToolRegistry()
    tools.register_defaults()
    agent = NestAgent(
        role="Agent Harness 团队",
        goal="完成用户交给 antNest Harness 的真实任务",
        backstory="精通 Agent Loop、Tool Use、上下文管理与沙箱执行。",
        llm=make_llm(), tools=tools)
    result = agent.kickoff("统计 /workspace/extracted 下 antNest team 职位文件数量，并生成报告。")
    print("Agent Loop 最终结果:", result)
    print("报告存在:", REPORT.exists(), "| 长期记忆: artifacts/memory.md")
    assert REPORT.exists(), "演示应产出报告文件"


if __name__ == "__main__":
    main()
