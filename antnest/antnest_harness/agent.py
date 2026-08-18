# -*- coding: utf-8 -*-
"""Agent 5：NestAgent —— antNest Harness 的核心 Agent Loop。

循环：系统提示(角色/目标/背景+工具说明) → LLM → 解析 JSON 动作
     → 工具执行(沙箱) → 结果回填 → 直到 finish 或 max_iter。
动作用 ```action 代码块包裹的 JSON 表达，兼容真实 LLM 与 MockLLM。
"""
import json
import re


class NestAgent:
    def __init__(self, role, goal, backstory, llm, tools, memory=None, max_iter=8):
        self.role, self.goal, self.backstory = role, goal, backstory
        self.llm, self.tools, self.max_iter = llm, tools, max_iter
        self.memory = memory

    def system_prompt(self) -> str:
        tool_doc = "\n".join(
            f'- {s["function"]["name"]}: {s["function"]["description"]}'
            for s in self.tools.schemas())
        return (
            f"你是 antNest 团队的「{self.role}」。\n目标：{self.goal}\n背景：{self.backstory}\n"
            f"可用工具：\n{tool_doc}\n"
            "每次回复只能是一个 ```action 代码块，内含 JSON：\n"
            '```action\n{"action":"tool","name":"工具名","args":{...}}\n```\n'
            "或任务完成时：\n"
            '```action\n{"action":"finish","result":"最终结果"}\n```\n')

    def kickoff(self, task: str) -> str:
        mem = self.memory
        if mem is None:
            from .memory import Memory
            mem = self.memory = Memory()
        mem.short_term = []
        mem.add("user", task)

        for _ in range(self.max_iter):
            reply = self.llm.complete(mem.messages(), tools=self.tools.schemas())
            mem.add("assistant", reply)
            action = self._parse(reply)
            if action is None:
                mem.add("tool", "[格式错误] 必须输出 ```action JSON 块")
                continue
            if action.get("action") == "finish":
                result = action.get("result", "")
                mem.note(f"[{self.role}] 完成: {result[:120]}")
                return result
            name, args = action.get("name"), action.get("args", {})
            obs = self.tools.execute(name, args)
            mem.add("tool", f"{name}({args}) -> {obs[:600]}")
        return f"达到最大迭代次数({self.max_iter})，任务中止。最近观察：{mem.short_term[-1]['content'][:200]}"

    @staticmethod
    def _parse(reply: str):
        m = re.search(r"```action\s*(\{.*?\})\s*```", reply, re.S)
        if not m:
            m = re.search(r"(\{.*?\})", reply, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
