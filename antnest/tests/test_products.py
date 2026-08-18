# -*- coding: utf-8 -*-
"""Agent 8（测试开发工程师）：antNest 产品线质量门禁。

覆盖：tokenizer 一致性 / 模型前向与训练收敛 / 沙箱安全 / Agent Loop / Multi-Agent。
"""
import json
from pathlib import Path

import pytest
import torch

from antnest_llm.tokenizer import CharTokenizer
from antnest_llm.model import TinyGPT
from antnest_llm.data import build_dataset, get_batch
from antnest_harness.llm import MockLLM
from antnest_harness.tools import ToolRegistry, _safe_path
from antnest_harness.agent import NestAgent
from antnest_harness.crew import NestCrew


# ── antNest LLM ────────────────────────────────────────────
def test_tokenizer_roundtrip():
    text = "蚁巢 antNest：训练大模型，打造 Harness。"
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text


def test_model_forward_shape():
    tok = CharTokenizer("蚁巢antNest")
    m = TinyGPT(len(tok), n_embd=32, n_head=2, n_layer=2, block_size=16)
    x = torch.randint(0, len(tok), (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, len(tok)) and loss.item() > 0


def test_training_loss_decreases():
    torch.manual_seed(0)
    tok = CharTokenizer("蚁巢模型与外殼训练数据" * 20)
    train_ids, val_ids = build_dataset(tok, block_size=32)
    m = TinyGPT(len(tok), n_embd=32, n_head=2, n_layer=2, block_size=32)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(8):
        x, y = get_batch(train_ids, val_ids, 32, 8)
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"训练应收敛: {losses[0]:.3f} -> {losses[-1]:.3f}"


# ── antNest Harness ────────────────────────────────────────
def test_sandbox_blocks_escape():
    with pytest.raises(PermissionError):
        _safe_path("/etc/passwd")


TMP = Path("/workspace/antnest/artifacts/test_tmp")


@pytest.fixture()
def tmpdir():
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def test_tools_write_read(tmpdir):
    reg = ToolRegistry(); reg.register_defaults()
    p = tmpdir / "x.txt"
    reg.execute("write_file", {"p": str(p), "c": "蚁巢"})
    assert "蚁巢" in reg.execute("read_file", {"p": str(p)})
    with pytest.raises(PermissionError):
        ToolRegistry._shell("rm -rf /")


def test_agent_loop_finish():
    reg = ToolRegistry(); reg.register_defaults()
    llm = MockLLM(script=['```action\n{"action":"finish","result":"完成"}\n```'])
    a = NestAgent("测试员", "验证 Agent Loop", "QA", llm=llm, tools=reg)
    assert a.kickoff("任意任务") == "完成"


def test_agent_loop_tool_then_finish(tmpdir):
    reg = ToolRegistry(); reg.register_defaults()
    out = tmpdir / "r.md"
    llm = MockLLM(script=[
        '```action\n{"action":"tool","name":"write_file","args":{"p":"%s","c":"ok"}}\n```' % out,
        '```action\n{"action":"finish","result":"已写入"}\n```'])
    a = NestAgent("执行师", "写文件", "Builder", llm=llm, tools=reg)
    assert a.kickoff("写文件") == "已写入" and out.read_text() == "ok"


def test_crew_multi_agent(tmpdir):
    reg = ToolRegistry(); reg.register_defaults()
    out = tmpdir / "crew.md"
    s = ['```action\n{"action":"finish","result":"计划：写入%s"}\n```' % out,
         '```action\n{"action":"tool","name":"write_file","args":{"p":"%s","c":"built"}}\n```' % out,
         '```action\n{"action":"finish","result":"built 已写入"}\n```',
         '```action\n{"action":"finish","result":"审查通过"}\n```']
    crew = NestCrew(llm=MockLLM(script=s), tools=reg)
    r = crew.kickoff("产出文件")
    assert r["review"] == "审查通过" and out.read_text() == "built"


def test_harness_artifacts_exist():
    art = Path("/workspace/antnest/artifacts")
    assert (art / "harness_demo_report.md").exists()
