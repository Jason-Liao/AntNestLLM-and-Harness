# -*- coding: utf-8 -*-
"""Agent 5 + Agent 6：工具层与沙箱执行。

Agent 6（Agent Infra / DSec）原则：所有工具在受限根目录内执行，
路径逃逸拒绝、shell 命令白名单 + 超时，构成进程级最小沙箱。
"""
import json
import shlex
import subprocess
from pathlib import Path

SAFE_ROOT = Path("/workspace").resolve()
ALLOWED_CMDS = {"ls", "cat", "wc", "head", "tail", "python", "pip", "grep", "find", "echo", "mkdir"}


def _safe_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = SAFE_ROOT / path
    path = path.resolve()
    if not str(path).startswith(str(SAFE_ROOT)):
        raise PermissionError(f"沙箱拒绝越界路径: {p}")
    return path


class ToolRegistry:
    """工具注册表：向 LLM 暴露 JSON Schema，向执行器暴露受控实现。"""

    def __init__(self):
        self._tools = {}

    def register(self, name, desc, func, params):
        self._tools[name] = {"desc": desc, "func": func, "params": params}

    # ── 内置工具 ────────────────────────────────────────────
    def register_defaults(self):
        self.register("list_dir", "列出目录下的文件名",
                      lambda p: json.dumps(
                          [x.name for x in _safe_path(p).iterdir()], ensure_ascii=False),
                      {"p": "目录路径"})
        self.register("read_file", "读取文本文件内容",
                      lambda p: _safe_path(p).read_text(encoding="utf-8")[:4000],
                      {"p": "文件路径"})
        self.register("write_file", "写入文本文件（覆盖）",
                      lambda p, c: (_safe_path(p).parent.mkdir(parents=True, exist_ok=True),
                                    str(_safe_path(p).write_text(c, encoding="utf-8")) + " bytes"),
                      {"p": "文件路径", "c": "内容"})
        self.register("shell", "在沙箱内执行白名单 shell 命令",
                      self._shell, {"cmd": "命令行"})
        # M6-4（Agent 4 检索专长）：参数化检索工具，动作空间 4 → 6
        self.register("grep", "在文件内容中搜索关键词，返回命中行",
                      self._grep, {"p": "文件路径", "q": "搜索关键词"})
        self.register("find", "按名称模式递归查找文件",
                      self._find, {"dir": "起始目录", "name": "文件名模式，如 *.py"})

    @staticmethod
    def _grep(p: str, q: str) -> str:
        """内容检索：沙箱内读文件，返回含关键词的行（上限 50 行）。"""
        hits = [ln.strip() for ln in _safe_path(p).read_text(
            encoding="utf-8", errors="ignore").splitlines() if q in ln]
        return json.dumps(hits[:50], ensure_ascii=False)

    @staticmethod
    def _find(dir: str, name: str) -> str:
        """名称检索：fnmatch 模式递归匹配（上限 200 条，惰性截断）。"""
        import fnmatch
        root = _safe_path(dir)
        hits = [str(x.relative_to(SAFE_ROOT)) for x in root.rglob("*")
                if fnmatch.fnmatch(x.name, name)][:200]
        return json.dumps(hits, ensure_ascii=False)

    @staticmethod
    def _shell(cmd: str) -> str:
        parts = shlex.split(cmd)
        if not parts or parts[0] not in ALLOWED_CMDS:
            raise PermissionError(f"命令不在白名单: {cmd}")
        r = subprocess.run(parts, capture_output=True, text=True, timeout=30, cwd=SAFE_ROOT)
        return (r.stdout + r.stderr)[:4000] or "(无输出)"

    # ── Schema / 执行 ───────────────────────────────────────
    def schemas(self):
        return [{"type": "function", "function": {"name": n,
                "description": t["desc"], "parameters": {"type": "object",
                "properties": t["params"]}}} for n, t in self._tools.items()]

    def execute(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"未知工具: {name}"
        try:
            return str(self._tools[name]["func"](**args))
        except Exception as e:  # 沙箱错误回传给 Agent 而非崩溃（Agent 6 容错要求）
            return f"[工具错误] {type(e).__name__}: {e}"
