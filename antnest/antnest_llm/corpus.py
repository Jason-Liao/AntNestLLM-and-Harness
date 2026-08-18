# -*- coding: utf-8 -*-
"""Agent 3（预训练数据工程师）：M1 语料扩展 v2。

在 32 份 JD（v1）基础上纳入：
  1. 33 份团队交付物（antnest_team/outputs/）
  2. antNest 产品源码（antnest_llm / antnest_harness）
实现"蚁巢用自己铸造自己"的数据自增长闭环。
"""
from pathlib import Path

from .tokenizer import load_corpus as load_corpus_v1

WS = Path("/workspace")


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_corpus_v2() -> str:
    parts = [load_corpus_v1()]
    for f in sorted((WS / "antnest_team" / "outputs").glob("*.md")):
        parts.append(f"《{f.stem}》\n" + _read(f))
    for pkg in ("antnest_llm", "antnest_harness"):
        for f in sorted((WS / "antnest" / pkg).glob("*.py")):
            parts.append(f"```python\n# {pkg}/{f.name}\n" + _read(f) + "\n```")
    return "\n\n".join(parts)


def load_corpus_v3() -> str:
    """清洗版 v2：去除 .doc 提取噪声（Agent 3，M2 数据质量升级）。"""
    from .cleaner import clean_text
    text, stats = clean_text(load_corpus_v2())
    globals()["V3_CLEAN_STATS"] = stats
    return text


def load_corpus_v4() -> str:
    """v3 + 外部合规语料（M3，Agent 3）。

    外部增量（corpus_extra.md）：公有领域先秦典籍 + PSF 许可证文本 +
    团队自写工程常识，全部无第三方版权风险。
    """
    extra = (WS / "antnest" / "corpus_extra.md").read_text(encoding="utf-8")
    return load_corpus_v3() + "\n\n" + extra


def jd_titles() -> list:
    """返回 [(编号, 职位名)]，供 SFT 构造职位问答。"""
    out = []
    for f in sorted((WS / "extracted").glob("*.md"),
                    key=lambda p: int(p.name.split(".")[0])):
        num, _, title = f.stem.partition(".")
        out.append((int(num), title.strip()))
    return out
