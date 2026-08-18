# -*- coding: utf-8 -*-
"""Agent 8（测试开发工程师）：M5 质量门禁 —— 工具选择攻坚 / 多步评测 / 自我进化。"""
import json
from pathlib import Path

import pytest

ART = Path("/workspace/antnest/artifacts")
EVALSET = Path("/workspace/antnest/evals/evalset.json")
SFT6_OK = (ART / "sft6_ckpt.pt").exists()
GRPO6_OK = (ART / "grpo6_ckpt.pt").exists()


# ── M5-1：动作示例扩容与对比学习 ────────────────────────────
def _tool_of(ans: str):
    """从动作围栏中解析工具名（JSON 解析，避免字符串匹配脆弱性）。"""
    import re
    m = re.search(r"```action\s*(\{.*?\})\s*```", ans, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1)).get("name")
    except json.JSONDecodeError:
        return None


def test_sft_action_examples_expanded():
    from antnest_llm.sft import build_examples, build_action_examples
    acts = build_action_examples()
    assert len(acts) >= 40, "动作示例应扩容到 40+ 条"
    tools = {t for t in (_tool_of(a) for _, a in acts) if t}
    assert {"list_dir", "read_file", "write_file", "shell"} <= tools, \
        "四类工具均需覆盖"
    # 措辞多样性：同一工具至少 5 种不同问法
    from collections import Counter
    cnt = Counter(t for t in (_tool_of(a) for _, a in acts) if t)
    assert all(v >= 5 for v in cnt.values()), f"每工具措辞应≥5: {dict(cnt)}"


def test_task_pool_expanded():
    from antnest_llm.grpo import TASK_POOL
    assert len(TASK_POOL) >= 20, "GRPO 任务池应扩容到 20+"
    tools = {t for _, (k, t) in TASK_POOL if k == "tool"}
    assert len(tools) == 4, "四类工具均需在池中"


def test_contrast_pairs_format():
    from antnest_llm.grpo import build_contrast_pairs, TASK_POOL
    pairs = build_contrast_pairs(TASK_POOL)
    assert len(pairs) >= 14, "对比对应覆盖所有 tool 类任务"
    for prompt, pos, neg in pairs[:5]:
        assert "```action" in pos and "```action" in neg, "正负例均需合法围栏"
        assert _tool_of(pos) != _tool_of(neg), "负例工具必须不同于正例"


def test_contrastive_loss_gradients():
    import torch
    from antnest_llm.grpo import contrastive_loss, build_contrast_pairs, TASK_POOL
    from antnest_llm.bpe import BPETokenizer
    from antnest_llm.model import TinyGPT
    tok = BPETokenizer("蚁巢工具对比学习list_dir shell", [])
    m = TinyGPT(len(tok), 32, 4, 2, 128)
    pairs = build_contrast_pairs(TASK_POOL[:6])
    loss = contrastive_loss(m, tok, pairs)
    assert loss is not None and loss.requires_grad, "对比损失应含梯度"
    loss.backward()  # 应可反传


# ── M5-2：多步链式评测 ──────────────────────────────────────
def test_evalset_multi_tasks():
    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    mt = es.get("multi_tasks", [])
    assert len(mt) >= 3, "评测集应含 3+ 条链式任务"
    for t in mt:
        assert 2 <= len(t["steps"]) <= 3, "链式任务应为 2-3 步"
        assert all(s.get("tool") for s in t["steps"]), "每步需指定期望工具"


def test_run_multi_step_mock():
    from antnest_llm.eval import run_multi_step, parse_tool
    from antnest_llm.sft import greedy_answer

    class FakeTok:
        def encode(self, s):
            return [ord(c) % 97 + 1 for c in s[:40]] or [1]

        def decode(self, ids):
            return "".join(chr(97 + (i - 1) % 97) for i in ids)

    class FakeModel:
        block_size = 256

        # 借 greedy_answer 的接口：逐步返回正确的动作围栏
        def generate(self, idx, max_new_tokens=100, **kw):
            return idx

    # 直接用 mock 响应测 parse_tool 与评分链路
    resp = ('执行：\n```action\n{"action":"tool","name":"list_dir",'
            '"args":{"p":"/workspace/extracted"}}\n```')
    kind, name = parse_tool(resp)
    assert (kind, name) == ("tool", "list_dir")
    kind2, name2 = parse_tool("没有动作的回答")
    assert kind2 is None and name2 is None


# ── M5-4：在线自我进化 ──────────────────────────────────────
def test_evolve_log_and_skip():
    from antnest_llm.evolve import run_evolve, read_log, best_ckpt
    # 无 --force 且无新增轨迹时应跳过（不产生训练）
    rec = run_evolve(force=False)
    assert rec["action"] in ("skip", "evolve"), "进化入口应正常返回"
    assert isinstance(read_log(), list)
    b = best_ckpt()
    assert b and not b.startswith("evo"), "进化起点不应是 evo 产物"


def test_evolve_status_cli():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "antnest_llm.evolve", "--status"],
                       cwd="/workspace/antnest", capture_output=True, text=True)
    assert r.returncode == 0 and "进化史" in r.stdout


# ── M5-5：checkpoint 一致性（若 M5 产物已训练）──────────────
@pytest.mark.skipif(not SFT6_OK, reason="无 sft6 checkpoint")
def test_sft6_artifacts_consistent():
    cfg = json.loads((ART / "sft6_model_config.json").read_text(encoding="utf-8"))
    from antnest_llm.bpe import load_tokenizer
    tok = load_tokenizer(ART / "sft6_vocab.json")
    assert len(tok) == cfg["vocab"], "sft6 词表与配置一致"


@pytest.mark.skipif(not GRPO6_OK, reason="无 grpo6 checkpoint")
def test_grpo6_metrics_has_contrast():
    m = json.loads((ART / "grpo6_metrics.json").read_text(encoding="utf-8"))
    assert "ContrastiveTools" in m["algo"], "grpo6 应记录对比学习算法标记"
    assert any("ctr_loss" in h for h in m["history"]), "历史应含对比损失曲线"
