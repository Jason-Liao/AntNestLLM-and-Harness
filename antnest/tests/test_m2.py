# -*- coding: utf-8 -*-
"""Agent 8（测试开发工程师）：M2 质量门禁 —— 清洗 / BPE / v3 一致性 / 对话模式。"""
import json
from pathlib import Path

import pytest

from antnest_llm.cleaner import clean_text
from antnest_llm.bpe import BPETokenizer, load_tokenizer
from antnest_llm.tokenizer import CharTokenizer

ART = Path("/workspace/antnest/artifacts")
V3_OK = (ART / "v3_ckpt.pt").exists() and (ART / "v3_model_config.json").exists()


def test_cleaner_drops_hex_noise():
    noise = "搀洀椀攀爀搀洀椀戀爀椀猀琀爀愀琀漀爀搀栀椀"
    text = "【团队使命】我们致力于 AGI 建设。\n" + noise + "\n负责大模型训练。"
    cleaned, stats = clean_text(text)
    assert "团队使命" in cleaned, "干净行应保留"
    assert noise not in cleaned, "十六进制噪声行应被剔除"
    assert stats["dropped_lines"] == 1


def test_bpe_roundtrip_and_compression():
    text = "蚁巢大模型训练管线：数据、清洗、分词、预训练、后训练。" * 20
    tok = BPETokenizer.train(text, vocab_size=len(set(text)) + 80)
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "训练文本应可无损往返"
    assert len(ids) < len(text), "合并应带来 token 压缩"
    assert all(0 <= i < len(tok) for i in ids)


def test_load_tokenizer_dispatch():
    d = ART / "test_tmp"
    d.mkdir(exist_ok=True, parents=True)
    bt = BPETokenizer("蚁巢训练", [("蚁", "巢")])
    bt.save(d / "b.json")
    assert isinstance(load_tokenizer(d / "b.json"), BPETokenizer)
    CharTokenizer("蚁巢模型").save(d / "c.json")
    assert isinstance(load_tokenizer(d / "c.json"), CharTokenizer)


@pytest.mark.skipif(not V3_OK, reason="无 v3 checkpoint（先运行 train --corpus v3 --tokenizer bpe）")
def test_v3_consistency_and_generate():
    import torch
    from antnest_llm.model import TinyGPT
    cfg = json.loads((ART / "v3_model_config.json").read_text(encoding="utf-8"))
    tok = load_tokenizer(ART / "v3_vocab.json")
    assert len(tok) == cfg["vocab"], "词表与配置一致"
    m = TinyGPT(cfg["vocab"], cfg["n_embd"], cfg["n_head"],
                cfg["n_layer"], cfg["block_size"])
    m.load_state_dict(torch.load(ART / "v3_ckpt.pt", weights_only=True))
    m.eval()
    ids = tok.encode("训练") or [0]
    out = m.generate(torch.tensor([ids], dtype=torch.long), max_new_tokens=24)
    assert out.shape[1] > len(ids), "v3 应能继续生成"


def test_chat_reply():
    from antnest_harness.chat import reply
    from antnest_harness.llm import AntNestLLMClient
    client = AntNestLLMClient()
    ans = reply(client, [], "antNest 的目标是什么？", max_new=40)
    assert isinstance(ans, str) and ans, "对话应返回非空回答"
