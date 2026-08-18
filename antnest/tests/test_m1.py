# -*- coding: utf-8 -*-
"""Agent 8（测试开发工程师）：M1 冲刺质量门禁。

覆盖：语料扩展 / SFT 掩码正确性 / antNest LLM 本地直连 Harness 端到端。
"""
from pathlib import Path

import pytest
import torch

from antnest_llm.tokenizer import CharTokenizer, load_corpus
from antnest_llm.corpus import load_corpus_v2
from antnest_llm.sft import encode_example, build_examples
from antnest_llm.model import TinyGPT

ART = Path("/workspace/antnest/artifacts")
HAS_CKPT = (ART / "v2_ckpt.pt").exists() and (ART / "v2_model_config.json").exists()


def test_corpus_v2_expanded():
    assert len(load_corpus_v2()) > len(load_corpus()), "v2 语料应大于 v1"


def test_sft_examples_and_masking():
    tok = CharTokenizer(load_corpus())
    ex = build_examples()
    assert len(ex) >= 36, "SFT 样例应覆盖 32 职位 + 知识 + 动作"
    q, a = ex[0]
    enc = encode_example(tok, q, a, block=512)
    assert enc is not None
    x, y = enc
    # prompt 区间（含 <|user|> 问题与 <|assistant|> 标记）应被 -100 掩码
    assert (y == -100).any(), "应存在掩码区间"
    sup = y[y != -100]
    assert len(sup) > 0 and len(sup) < len(y), "仅监督 assistant 部分"
    # 被监督 token 解码后应出现在答案文本中（字符级子集）
    assert set(tok.decode(sup.tolist())) <= set(a) | set(q)


def test_generate_handles_long_prompt():
    tok = CharTokenizer("蚁巢模型" * 200)
    m = TinyGPT(len(tok), n_embd=32, n_head=2, n_layer=2, block_size=32)
    idx = torch.randint(0, len(tok), (1, 100))  # 超过 ctx 的 prompt
    out = m.generate(idx, max_new_tokens=5)
    assert out.shape[1] == 105, "长 prompt 应被截断到 ctx 内且可继续生成"


@pytest.mark.skipif(not HAS_CKPT, reason="无 v2 checkpoint（先运行 train --corpus v2）")
def test_antnest_llm_direct_connection():
    """模型-Harness 共同进化闭环：本地 antNest LLM 驱动 Agent Loop 完成任务。"""
    from antnest_harness.llm import AntNestLLMClient
    from antnest_harness.tools import ToolRegistry
    from antnest_harness.agent import NestAgent

    tools = ToolRegistry(); tools.register_defaults()
    client = AntNestLLMClient()
    agent = NestAgent("Harness 团队", "完成直连验证任务", "M1", 
                      llm=client, tools=tools, max_iter=6)
    result = agent.kickoff("检查职位语料目录并生成 M1 直连验证报告")
    assert "antNest" in result or "任务" in result or "完成" in result
    assert (ART / "harness_m1_report.md").exists(), "直连任务应产出报告"


@pytest.mark.skipif(not HAS_CKPT, reason="无 v2 checkpoint")
def test_ckpt_vocab_consistency():
    """词表版本锁定：checkpoint 词表尺寸 == 配置文件 vocab。"""
    import json
    cfg = json.loads((ART / "v2_model_config.json").read_text(encoding="utf-8"))
    tok = CharTokenizer.load(str(ART / "v2_vocab.json"))
    assert len(tok) == cfg["vocab"]
