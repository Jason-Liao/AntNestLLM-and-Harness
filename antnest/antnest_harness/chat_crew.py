# -*- coding: utf-8 -*-
"""Agent 5 + Agent 7 + Agent 24：对话模式接入 Multi-Agent（M3，M4 轨迹回流）。

ChatCrew = 对话路由层：
  - 任务型消息（含动作意图）→ NestCrew 三智能体闭环（规划→执行→审查），
    由本地 antNest LLM 驱动，全程沙箱工具
  - 闲聊/知识型消息 → chat.reply 轻量直答
对话记录追加至 artifacts/chat_log.md。

M4 升级（轨迹回流）：
  - TrajRecorder 包装 ToolRegistry，记录每次真实工具调用 (name, args, ok)
  - Crew 收敛且审查通过 → 轨迹写入 artifacts/trajs.jsonl，
    供 SFT --traj 回流训练（"使用即训练"数据闭环）

运行：PYTHONPATH=/workspace/antnest python -m antnest_harness.chat_crew --message "统计交付物数量并写报告"
"""
import argparse
import json
from pathlib import Path

from .chat import reply as chat_reply
from .crew import NestCrew
from .llm import AntNestLLMClient
from .tools import ToolRegistry

LOG = Path("/workspace/antnest/artifacts/chat_log.md")
TRAJ = Path("/workspace/antnest/artifacts/trajs.jsonl")

# 任务型意图词（命中任意即走 Crew；评测集措辞刻意不同，防过拟合指标污染路由）
# M6-4：新增检索类动词（检索/搜/查找/找），覆盖 grep / find 工具任务
TASK_HINTS = ("列出", "统计", "写", "保存", "读取", "查看", "删除",
              "list", "count", "write", "read", "报告", "清单", "多少",
              "看看", "列一下", "都有什么", "念念", "到此为止", "收工",
              "检索", "搜", "查找", "找")


def is_task(msg: str) -> bool:
    return any(h in msg for h in TASK_HINTS)


class TrajRecorder:
    """Agent 7：包装工具注册表，透明记录每次调用（M4 轨迹回流）。"""

    def __init__(self, inner: ToolRegistry):
        self.inner = inner
        self.calls: list = []

    def __getattr__(self, name):  # schemas() 等透传
        return getattr(self.inner, name)

    def execute(self, name: str, args: dict) -> str:
        obs = self.inner.execute(name, args)
        self.calls.append({"name": name, "args": args,
                           "ok": not obs.startswith(("[工具错误]", "未知工具"))})
        return obs


def _dump_traj(task: str, calls: list, ok: bool):
    TRAJ.parent.mkdir(exist_ok=True)
    with TRAJ.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"task": task, "calls": calls, "ok": ok},
                           ensure_ascii=False) + "\n")


def run(client: AntNestLLMClient, message: str) -> dict:
    """一条消息 → 路由 → Crew 协作或轻量直答，返回结构化回合。"""
    if not is_task(message):
        answer = chat_reply(client, [], message)
        return {"route": "chat", "answer": answer, "crew": None}

    tools = TrajRecorder(ToolRegistry())
    tools.register_defaults()
    crew = NestCrew(llm=client, tools=tools)
    out = crew.kickoff(message)
    review = out.get("review", "")
    converged = "中止" not in review and "最大迭代" not in review
    # 轨迹回流：收敛回合入库（成功轨迹供 SFT 回流；失败轨迹保留供 badcase 分析）
    _dump_traj(message, tools.calls, converged)
    return {"route": "crew", "answer": review,
            "crew": {"plan": out.get("plan", "")[:300],
                     "built": out.get("built", "")[:300],
                     "review": review[:300]},
            "traj_calls": len(tools.calls)}


def read_batch_tasks(path: str) -> list:
    """M7-4：读取批量任务文件（每行一条，# 开头为注释，去空行）。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines
            if ln.strip() and not ln.strip().startswith("#")]


def run_batch(client, messages: list) -> dict:
    """M7-4：批量任务模式——一条命令积累一批轨迹（轨迹积累提速）。

    进化闭环的瓶颈在轨迹积累速度：逐条对话喂任务太慢，批量模式
    一次跑 N 条任务型消息，成功轨迹连续入库 artifacts/trajs.jsonl，
    供 evolve.py 检测新轨迹触发增量进化。
    """
    stats = {"crew": 0, "chat": 0, "trajs": 0}
    for i, msg in enumerate(messages, 1):
        turn = run(client, msg)
        route = turn["route"]
        stats[route] = stats.get(route, 0) + 1
        if route == "crew":
            stats["trajs"] += 1
        print(f"[{i}/{len(messages)}] {route.upper():4} | 工具调用 "
              f"{turn.get('traj_calls', 0)} 次 | {msg[:40]}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="antNest 对话 × Multi-Agent")
    ap.add_argument("--message", default="统计团队交付物数量并写成报告")
    ap.add_argument("--batch", default=None,
                    help="M7-4 批量任务文件：每行一条消息，轨迹批量积累")
    args = ap.parse_args()

    client = AntNestLLMClient()
    print(f"antNest ChatCrew | checkpoint: {client.ckpt_name} | "
          f"参数 {sum(p.numel() for p in client.model.parameters())/1e6:.1f}M")

    if args.batch:
        messages = read_batch_tasks(args.batch)
        print(f"批量模式：{len(messages)} 条任务 → 轨迹批量积累\n")
        stats = run_batch(client, messages)
        print(f"\n完成：Crew {stats['crew']} 条 / Chat {stats['chat']} 条 | "
              f"轨迹 +{stats['trajs']} 条 → {TRAJ}")
        return

    turn = run(client, args.message)
    print(f"\n路由: {turn['route'].upper()}")
    if turn["crew"]:
        print(f"规划: {turn['crew']['plan'][:150]}")
        print(f"执行: {turn['crew']['built'][:150]}")
        print(f"审查: {turn['crew']['review'][:150]}")
        print(f"工具调用: {turn.get('traj_calls', 0)} 次 → {TRAJ}")
    print(f"\n蚁巢: {turn['answer'][:300]}")

    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"**用户**：{args.message}\n\n**蚁巢({turn['route']}）**："
                f"{turn['answer'][:400]}\n\n---\n\n")
    print(f"\n已记录: {LOG}")


if __name__ == "__main__":
    main()
