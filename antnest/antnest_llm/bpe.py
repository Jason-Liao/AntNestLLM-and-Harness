# -*- coding: utf-8 -*-
"""Agent 3 + Agent 24：BPE tokenizer —— M2 词表升级。

以字符为初始符号，在"行级词频表"上迭代合并最高频相邻对；
中文友好（无需预分词），接口与 CharTokenizer 对齐
（encode / decode / save / load / __len__）。
"""
import json
from pathlib import Path


class BPETokenizer:
    def __init__(self, base=(), merges=()):
        self.base = list(base)
        self.merges = [tuple(m) for m in merges]
        self._base_set = set(self.base)
        self._rank = {p: i for i, p in enumerate(self.merges)}
        self._id_of = {c: i for i, c in enumerate(self.base)}
        for i, (a, b) in enumerate(self.merges):
            self._id_of[a + b] = len(self.base) + i
        self._piece_of = {i: p for p, i in self._id_of.items()}

    def __len__(self):
        return len(self.base) + len(self.merges)

    # ── 训练 ─────────────────────────────────────────────
    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        freq: dict = {}
        for line in text.split("\n"):
            line = line.strip()
            if line:
                key = tuple(line)
                freq[key] = freq.get(key, 0) + 1
        base = sorted(set(text) - {"\n"})
        n_merge = max(0, vocab_size - len(base))
        merges, words = [], dict(freq)
        for _ in range(n_merge):
            pairs: dict = {}
            for w, c in words.items():
                for i in range(len(w) - 1):
                    p = (w[i], w[i + 1])
                    pairs[p] = pairs.get(p, 0) + c
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break
            merges.append(best)
            nb = best[0] + best[1]
            words = {cls._apply(w, best, nb): c for w, c in words.items()}
        return cls(base, merges)

    @staticmethod
    def _apply(word: tuple, pair: tuple, merged: str) -> tuple:
        out, i, w = [], 0, list(word)
        while i < len(w):
            if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
                out.append(merged)
                i += 2
            else:
                out.append(w[i])
                i += 1
        return tuple(out)

    # ── 编解码（rank 贪心，等价于按合并次序应用）─────────
    def _encode_line(self, line: str) -> list:
        parts = [c for c in line if c in self._base_set]
        while len(parts) > 1:
            best_i, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self._rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        return [self._id_of[p] for p in parts]

    def encode(self, s: str) -> list:
        ids = []
        for line in s.split("\n"):
            line = line.strip()
            if line:
                ids += self._encode_line(line)
        return ids

    def decode(self, ids: list) -> str:
        return "".join(self._piece_of.get(i, "") for i in ids)

    # ── 持久化 ───────────────────────────────────────────
    def save(self, path):
        Path(path).write_text(json.dumps(
            {"type": "bpe", "base": self.base,
             "merges": [list(m) for m in self.merges]}, ensure_ascii=False),
            encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["base"], d["merges"])


def load_tokenizer(path):
    """按文件内容自动识别并加载 Char / BPE tokenizer（版本无关）。"""
    from .tokenizer import CharTokenizer
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("type") == "bpe":
        return BPETokenizer(d["base"], d["merges"])
    tok = CharTokenizer()
    tok.stoi = {c: int(i) for c, i in d["stoi"].items()}
    tok.itos = {i: c for c, i in tok.stoi.items()}
    return tok
