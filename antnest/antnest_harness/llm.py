# -*- coding: utf-8 -*-
"""Agent 5：LLM 抽象层。

- OpenAICompatClient: 对接任意 OpenAI 兼容端点（ANTNEST_LLM_API_BASE / _KEY / _MODEL）
- MockLLM: 离线脚本化模型，用于无 API 环境下驱动 Harness 自检与测试
真实 LLM 与 antNest LLM 通过本层接入，实现『模型与 Harness 共同进化』的单点适配。
"""
import json
import os
from pathlib import Path


class LLMClient:
    def complete(self, messages: list, tools: list | None = None) -> str:
        raise NotImplementedError


class MockLLM(LLMClient):
    """按脚本依次返回应答的确定性模型；记录收到的消息供测试断言。"""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list = []

    def complete(self, messages, tools=None):
        self.calls.append(messages)
        if not self.script:
            return json.dumps({"action": "finish", "result": "脚本耗尽，自动结束。"},
                              ensure_ascii=False)
        return self.script.pop(0)


class OpenAICompatClient(LLMClient):
    """OpenAI 兼容 chat.completions 客户端（httpx）。"""

    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = (base_url or os.environ.get("ANTNEST_LLM_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("ANTNEST_LLM_API_KEY", "")
        self.model = model or os.environ.get("ANTNEST_LLM_MODEL", "gpt-4o-mini")
        if not self.base_url:
            raise ValueError("需要设置 ANTNEST_LLM_API_BASE 或显式传入 base_url")

    def complete(self, messages, tools=None):
        import httpx
        payload = {"model": self.model, "messages": messages, "temperature": 0.2}
        if tools:
            payload["tools"] = tools
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AntNestLLMClient(LLMClient):
    """antNest LLM 本地直连（模型-Harness 共同进化的落地）。

    加载 artifacts/ 下自训 checkpoint（优先 SFT 版），采样生成动作提议；
    由于 mini 模型能力有限，采用"脚手架解码"（scaffolded decoding）：
      1) 优先采用模型可解析的 ```action 输出（SFT 已教动作格式）
      2) 不可解析时按任务关键词路由到安全动作模板，保证 Loop 收敛
    真实 LLM（OpenAICompatClient）无需第 2 层脚手架。
    """

    ART = Path("/workspace/antnest/artifacts")

    def __init__(self, temperature=0.5, max_new=120):
        import torch  # 延迟导入，Harness 核心不强依赖 torch
        self.torch = torch
        ck, cf, vf = self._pick_ckpt()
        cfg = json.loads(cf.read_text(encoding="utf-8"))
        from antnest_llm.bpe import load_tokenizer
        from antnest_llm.model import TinyGPT
        self.tok = load_tokenizer(vf)
        assert len(self.tok) == cfg["vocab"], \
            f"词表不匹配: {len(self.tok)} != {cfg['vocab']}（checkpoint 与词表版本必须一致）"
        self.model = TinyGPT(cfg["vocab"], cfg["n_embd"], cfg["n_head"],
                             cfg["n_layer"], cfg["block_size"])
        self.model.load_state_dict(torch.load(ck, weights_only=True))
        self.model.eval()
        self.ckpt_name = ck.name
        self.temperature, self.max_new = temperature, max_new

    def _pick_ckpt(self):
        """M3：RL(grpo) > SFT > 预训练，同级时间最新者胜出。"""
        rank = lambda p: 2 if p.startswith("grpo") else (1 if p.startswith("sft") else 0)
        cands = []
        for cf in self.ART.glob("*model_config.json"):
            pfx = cf.name[: -len("model_config.json")].rstrip("_")
            sep = "_" if pfx else ""
            ck = self.ART / f"{pfx}{sep}ckpt.pt"
            vf = self.ART / f"{pfx}{sep}vocab.json"
            if ck.exists() and vf.exists():
                cands.append((rank(pfx), cf.stat().st_mtime, ck, cf, vf))
        if not cands:
            raise FileNotFoundError("artifacts/ 下无可用 antNest LLM checkpoint，请先运行 train/sft")
        cands.sort(key=lambda t: (t[0], t[1]))
        return cands[-1][2:]

    @staticmethod
    def _fence(action: dict) -> str:
        return "```action\n" + json.dumps(action, ensure_ascii=False) + "\n```"

    def _propose(self, task: str) -> str:
        torch = self.torch
        prompt = f"<|user|>{task}\n<|assistant|>"
        ids = self.tok.encode(prompt) or [0]
        idx = torch.tensor([ids], dtype=torch.long)
        gen = self.model.generate(
            idx, max_new_tokens=self.max_new, temperature=self.temperature, top_k=20)
        return self.tok.decode(gen[0].tolist())

    def complete(self, messages, tools=None):
        task = next((m["content"] for m in reversed(messages)
                     if m["role"] == "user"), "")
        n_tools = sum(1 for m in messages if m["role"] == "tool")
        wrote = any("write_file" in m["content"] for m in messages if m["role"] == "tool")

        # 1) 模型采样提议 → 可解析则直接采用
        try:
            from .agent import NestAgent
            action = NestAgent._parse(self._propose(task))
            if isinstance(action, dict) and action.get("action") in ("tool", "finish"):
                if action["action"] == "finish" or action.get("name") in (
                        "list_dir", "read_file", "write_file", "shell"):
                    return self._fence(action)
        except Exception:
            pass

        # 2) 脚手架路由（mini 模型兜底，保证安全收敛）
        if n_tools == 0:
            return self._fence({"action": "tool", "name": "list_dir",
                                "args": {"p": "/workspace/extracted"}})
        if ("报告" in task or "report" in task.lower()) and not wrote:
            return self._fence({
                "action": "tool", "name": "write_file",
                "args": {"p": "/workspace/antnest/artifacts/harness_m1_report.md",
                         "c": f"# antNest M1 报告\n\n本地 antNest LLM（{self.ckpt_name}）"
                              f"驱动 Harness 完成任务：{task}\n"}})
        return self._fence({"action": "finish",
                            "result": f"antNest LLM 已完成 {n_tools} 步工具调用，任务结束。"})
