# -*- coding: utf-8 -*-
"""Agent 3（预训练数据工程师）：语料清洗器 —— M2 数据质量升级。

背景：32 份 .doc 提取文本中残留 UTF-16 十六进制噪声（如 搀洀椀攀爀，
为 ASCII 字节的误读映射），M0 采样中已实际污染输出。
方法：以团队交付物（人工撰写、零噪声）构建"干净字符集"，
行内干净集外非 ASCII 字符占比 > 15% 判为噪声行剔除。
"""
from pathlib import Path

WS = Path("/workspace")
NOISE_RATIO = 0.15

_CLEAN: set | None = None


def _clean_set() -> set:
    chars = set()
    for f in (WS / "antnest_team" / "outputs").glob("*.md"):
        chars |= set(f.read_text(encoding="utf-8"))
    chars |= set("，。、；：？！（）【】《》""''—…·0123456789 ")
    return chars


def clean_text(text: str, clean=None) -> tuple:
    """返回 (清洗后文本, 统计)。纯 ASCII 行与空行直接保留。"""
    global _CLEAN
    if _CLEAN is None:
        _CLEAN = clean or _clean_set()
    kept, dropped, dchars = [], 0, 0
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.isascii():
            kept.append(line)
            continue
        bad = sum(1 for ch in line if not ch.isascii() and ch not in _CLEAN)
        if bad / len(line) > NOISE_RATIO:
            dropped += 1
            dchars += len(line)
        else:
            kept.append(line)
    stats = {"dropped_lines": dropped, "dropped_chars": dchars,
             "kept_chars": len(text) - dchars, "total_chars": len(text)}
    return "\n".join(kept), stats
