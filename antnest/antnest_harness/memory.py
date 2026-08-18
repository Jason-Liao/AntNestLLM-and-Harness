# -*- coding: utf-8 -*-
"""Agent 5：记忆与上下文管理。

- 短期记忆：本轮任务的对话轮次
- 长期记忆：跨任务持久化笔记（artifacts/memory.md）
- 上下文管理：超出窗口时压缩最早的工具结果（保留 system 与最近 N 轮）
"""
from pathlib import Path
import time

MEMORY_FILE = Path("/workspace/antnest/artifacts/memory.md")


class Memory:
    def __init__(self, path: Path = MEMORY_FILE):
        self.path = path
        self.short_term: list = []

    # ── 短期 ──
    def add(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content, "ts": time.time()})

    def messages(self, max_turns: int = 24) -> list:
        """上下文管理：保留全部非工具消息 + 最近若干轮工具消息。"""
        msgs = [m for m in self.short_term if m["role"] != "tool"]
        tools = [m for m in self.short_term if m["role"] == "tool"]
        keep = msgs + tools[-max_turns:]
        return [{"role": m["role"], "content": m["content"]} for m in keep]

    # ── 长期 ──
    def note(self, text: str):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"- {time.strftime('%F %T')} {text}\n")

    def long_term(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.exists() else ""
