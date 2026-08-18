# -*- coding: utf-8 -*-
"""Agent 5：NestCrew —— Subagent / Multi-Agent 协作。

规划者拆解任务 → 执行者调用工具完成 → 审查者校验产出，
即 antNest Harness 的最小多智能体闭环（planner → builder → reviewer）。
"""
from pathlib import Path
from .agent import NestAgent
from .memory import Memory

PLANNER = ("任务规划师", "把用户任务拆解为一步步可执行的工具调用计划",
           "antNest Harness 规划 Subagent，擅长结构化拆解。")
BUILDER = ("任务执行师", "严格按计划调用工具完成实际工作",
           "antNest Harness 执行 Subagent，擅长使用工具。")
REVIEWER = ("质量审查师", "校验执行结果是否达成任务目标",
            "antNest Harness 审查 Subagent，擅长发现遗漏。")


class NestCrew:
    def __init__(self, llm, tools, plan_role=PLANNER, build_role=BUILDER, review_role=REVIEWER):
        self.memory = Memory()
        self.planner = NestAgent(*plan_role, llm=llm, tools=tools, memory=self.memory)
        self.builder = NestAgent(*build_role, llm=llm, tools=tools, memory=self.memory)
        self.reviewer = NestAgent(*review_role, llm=llm, tools=tools, memory=self.memory)

    def kickoff(self, task: str) -> dict:
        plan = self.planner.kickoff(task)
        built = self.builder.kickoff(f"按以下计划完成任务「{task}」：\n{plan}")
        review = self.reviewer.kickoff(f"任务「{task}」的执行结果：\n{built}\n请审查是否达成目标。")
        return {"plan": plan, "built": built, "review": review}
