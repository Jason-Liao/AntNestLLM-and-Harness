# -*- coding: utf-8 -*-
"""Agent 3（预训练数据工程师）：字符级 tokenizer，语料取自 antNest team 32 个职位描述。"""
import json
from pathlib import Path

CORPUS_DIR = Path("/workspace/extracted")


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> str:
    """采集 32 个职位描述文本，拼接为预训练语料（含文件标题分隔行）。"""
    parts = []
    for f in sorted(corpus_dir.glob("*.md"), key=lambda p: int(p.name.split(".")[0])):
        parts.append(f"《{f.stem}》\n" + f.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


class CharTokenizer:
    """字符级词表：构建 / 编码 / 解码 / 持久化。"""

    def __init__(self, text: str = ""):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        if not text:  # 空词表占位
            self.stoi, self.itos = {}, {}

    def __len__(self):
        return len(self.stoi)

    def encode(self, s: str) -> list:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def save(self, path):
        Path(path).write_text(
            json.dumps({"stoi": self.stoi}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CharTokenizer":
        tok = cls()
        tok.stoi = {c: int(i) for c, i in json.loads(Path(path).read_text(encoding="utf-8"))["stoi"].items()}
        tok.itos = {i: c for c, i in tok.stoi.items()}
        return tok
