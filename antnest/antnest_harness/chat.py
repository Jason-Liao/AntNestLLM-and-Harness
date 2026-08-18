# -*- coding: utf-8 -*-
"""Agent 5 + Agent 7：antNest LLM 对话模式（M2）。

本地自训模型的多轮 Chat：SFT 模板 <|user|>/<|assistant|> 拼接历史，
截断到模型 ctx 后采样生成；对话记录追加至 artifacts/chat_log.md。
运行：PYTHONPATH=/workspace/antnest python -m antnest_harness.chat --message "问题"
"""
import argparse
from pathlib import Path

from .llm import AntNestLLMClient

LOG = Path("/workspace/antnest/artifacts/chat_log.md")


def reply(client, history: list, user_text: str,
          max_new: int = 80, temperature: float = 0.7) -> str:
    """一轮对话：拼接历史 → 截断 ctx → 采样 → 取新增 token 解码。"""
    prompt = "".join(f"<|user|>{q}\n<|assistant|>{a}\n" for q, a in history)
    prompt += f"<|user|>{user_text}\n<|assistant|>"
    ids = client.tok.encode(prompt) or [0]
    ctx = client.model.block_size
    idx = client.torch.tensor([ids[-ctx:]], dtype=client.torch.long)
    out = client.model.generate(
        idx, max_new_tokens=max_new, temperature=temperature, top_k=20)
    new = out[0].tolist()[idx.shape[1]:]
    return client.tok.decode(new).strip() or "……"


def main():
    ap = argparse.ArgumentParser(description="antNest LLM 本地对话")
    ap.add_argument("--message", default="antNest 的目标是什么？")
    ap.add_argument("--max_new", type=int, default=80)
    args = ap.parse_args()

    client = AntNestLLMClient()
    print(f"antNest LLM 对话模式 | checkpoint: {client.ckpt_name} "
          f"| 词表 {len(client.tok)} | 参数 {sum(p.numel() for p in client.model.parameters())/1e3:.0f}K")

    history = []
    for q in [args.message, "antNest Harness 有什么能力？"]:
        a = reply(client, history, q, max_new=args.max_new)
        history.append((q, a))
        print(f"\n用户: {q}\n蚁巢: {a}")

    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        for q, a in history:
            f.write(f"**用户**：{q}\n\n**蚁巢**：{a}\n\n---\n\n")
    print(f"\n对话已记录: {LOG}")


if __name__ == "__main__":
    main()
