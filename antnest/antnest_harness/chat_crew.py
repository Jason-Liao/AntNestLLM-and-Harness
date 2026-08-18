# -*- coding: utf-8 -*-
"""Agent 5 + Agent 7 + Agent 24：对话模式接入 Multi-Agent（M3）。

ChatCrew = 对话路由层：
  - 任务型消息（含动作意图）→ NestCrew 三智能体闭环（规划→执行→审查），
    由本地 antNest LLM 驱动，全程沙箱工具
  - 闲聊/知识型消息 → chat.reply 轻量直答
对话记录追加至 artifacts/chat_log.md。

运行：PYTHONPATH=/workspace/antnest python -m antnest_harness.chat_crew --message "统计交付物数量并写报告"
"""
import argparse
from pathlib import Path

from .chat import reply as chat_reply
from .crew import NestCrew
from .llm import AntNestLLMClient
from .tools import ToolRegistry

LOG = Path("/workspace/antnest/artifacts/chat_log.md")

# 任务型意图词（命中任意即走 Crew；评测集措辞刻意不同，防过拟合指标污染路由）
TASK_HINTS = ("列出", "统计", "写", "保存", "读取", "查看", "删除",
              "list", "count", "write", "read", "报告", "清单", "多少",
              "看看", "列一下", "都有什么", "念念", "到此为止", "收工")


def is_task(msg: str) -> bool:
    return any(h in msg for h in TASK_HINTS)


def run(client: AntNestLLMClient, message: str) -> dict:
    """一条消息 → 路由 → Crew 协作或轻量直答，返回结构化回合。"""
    if not is_task(message):
        answer = chat_reply(client, [], message)
        return {"route": "chat", "answer": answer, "crew": None}

    tools = ToolRegistry()
    tools.register_defaults()
    crew = NestCrew(llm=client, tools=tools)
    out = crew.kickoff(message)
    return {"route": "crew", "answer": out.get("review", ""),
            "crew": {"plan": out.get("plan", "")[:300],
                     "built": out.get("built", "")[:300],
                     "review": out.get("review", "")[:300]}}


def main():
    ap = argparse.ArgumentParser(description="antNest 对话 × Multi-Agent")
    ap.add_argument("--message", default="统计团队交付物数量并写成报告")
    args = ap.parse_args()

    client = AntNestLLMClient()
    print(f"antNest ChatCrew | checkpoint: {client.ckpt_name} | "
          f"参数 {sum(p.numel() for p in client.model.parameters())/1e6:.1f}M")
    turn = run(client, args.message)
    print(f"\n路由: {turn['route'].upper()}")
    if turn["crew"]:
        print(f"规划: {turn['crew']['plan'][:150]}")
        print(f"执行: {turn['crew']['built'][:150]}")
        print(f"审查: {turn['crew']['review'][:150]}")
    print(f"\n蚁巢: {turn['answer'][:300]}")

    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"**用户**：{args.message}\n\n**蚁巢({turn['route']}）**："
                f"{turn['answer'][:400]}\n\n---\n\n")
    print(f"\n已记录: {LOG}")


if __name__ == "__main__":
    main()
