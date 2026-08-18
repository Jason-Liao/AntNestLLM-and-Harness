# -*- coding: utf-8 -*-
"""Agent 8（测试开发工程师）：M4 质量门禁 —— PRM / 对齐税锚定 / 轨迹回流 / 评测扩容。"""
import json
from pathlib import Path

import pytest
import torch

from antnest_llm.grpo import (evaluate_response, sandbox_ok, sft_anchor_loss,
                              TASK_POOL, TOOL_ARGS)
from antnest_llm.sft import load_traj_examples
from antnest_llm.eval import score_action, score_qa, PASS_BAR, EVALSET
from antnest_harness.chat_crew import TrajRecorder, is_task
from antnest_harness.tools import ToolRegistry

ART = Path("/workspace/antnest/artifacts")
GRPO5_OK = (ART / "grpo5_ckpt.pt").exists()


def _fence(act: dict) -> str:
    return "```action\n" + json.dumps(act, ensure_ascii=False) + "\n```"


# ── M4-① PRM 五级过程奖励 ──────────────────────────────────────
def test_prm_l3_tool_selection():
    right = _fence({"action": "tool", "name": "list_dir", "args": {"p": "/workspace"}})
    wrong = _fence({"action": "tool", "name": "shell", "args": {"cmd": "ls"}})
    exp = ("tool", "list_dir")
    assert evaluate_response(right, exp) > evaluate_response(wrong, exp), \
        "选对工具（L3 0.3）必须显著高于选错工具"


def test_prm_l4_param_keys():
    full = _fence({"action": "tool", "name": "write_file",
                   "args": {"p": "/workspace/antnest/artifacts/t.md", "c": "x"}})
    miss = _fence({"action": "tool", "name": "write_file", "args": {"p": "/tmp/x"}})
    exp = ("tool", "write_file")
    assert evaluate_response(full, exp) - evaluate_response(miss, exp) >= 0.15


def test_prm_l5_sandbox_execution():
    ok = _fence({"action": "tool", "name": "list_dir", "args": {"p": "/workspace"}})
    bad = _fence({"action": "tool", "name": "list_dir", "args": {"p": "/etc"}})
    exp = ("tool", "list_dir")
    assert evaluate_response(ok, exp) > evaluate_response(bad, exp), \
        "沙箱执行失败（路径越界）不应拿 L5 分"


def test_sandbox_ok_direct():
    assert sandbox_ok({"action": "tool", "name": "shell", "args": {"cmd": "ls /workspace"}})
    assert not sandbox_ok({"action": "tool", "name": "shell", "args": {"cmd": "rm -rf /"}})
    assert sandbox_ok({"action": "finish"})


def test_tool_args_table():
    assert TOOL_ARGS["write_file"] == {"p", "c"}
    for _, (kind, tool) in TASK_POOL:
        if kind == "tool":
            assert tool in TOOL_ARGS, f"期望工具 {tool} 必须有参数键定义"


def test_anchor_loss_runs():
    from antnest_llm.model import TinyGPT
    from antnest_llm.tokenizer import CharTokenizer
    tok = CharTokenizer("蚁巢模型锚定问答")
    m = TinyGPT(len(tok), 16, 2, 1, 32)
    loss = sft_anchor_loss(m, tok, [("蚁巢是什么？", "蚁巢是大模型计划。")], 32)
    assert loss is not None and torch.isfinite(loss)


# ── M4-② 轨迹回流 ─────────────────────────────────────────────
def test_load_traj_examples(tmp_path):
    f = tmp_path / "trajs.jsonl"
    f.write_text("\n".join([
        json.dumps({"task": "列出目录", "ok": True,
                    "calls": [{"name": "list_dir", "args": {"p": "/workspace"}, "ok": True}]},
                   ensure_ascii=False),
        json.dumps({"task": "失败任务", "ok": False, "calls": []}),
    ]), encoding="utf-8")
    ex = load_traj_examples(str(f))
    assert len(ex) == 1 and ex[0][0] == "列出目录"
    assert "list_dir" in ex[0][1] and "```action" in ex[0][1]


def test_traj_recorder_wraps_registry():
    rec = TrajRecorder(ToolRegistry())
    rec.register_defaults()
    obs = rec.execute("list_dir", {"p": "/workspace/antnest"})
    assert not obs.startswith("[工具错误]")
    assert len(rec.calls) == 1 and rec.calls[0]["ok"] is True
    rec.execute("list_dir", {"p": "/etc"})
    assert rec.calls[1]["ok"] is False, "越界路径调用应记为失败"
    assert len(rec.schemas()) >= 4, "__getattr__ 透传 schemas() 等方法"


# ── M4-③ 评测扩容 + pass@k 尺子 ─────────────────────────────────
def test_evalset_expanded():
    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    assert len(es["action_tasks"]) >= 20
    assert len(es["qa_tasks"]) >= 10
    train_prompts = {p for p, _ in TASK_POOL}
    eval_prompts = {t["prompt"] for t in es["action_tasks"]}
    assert not (train_prompts & eval_prompts), "评测任务不得与训练任务同措辞"
    tools = {t["tool"] for t in es["action_tasks"] if t["expect"] == "tool"}
    # M6-4：动作空间扩至 6 工具（+grep/find）
    assert tools == {"list_dir", "shell", "write_file", "read_file", "grep", "find"}


def test_score_action_tool_grade():
    right = _fence({"action": "tool", "name": "list_dir", "args": {"p": "/workspace"}})
    wrong = _fence({"action": "tool", "name": "read_file", "args": {"p": "/workspace"}})
    assert score_action(right, "tool", "list_dir") >= PASS_BAR
    assert score_action(wrong, "tool", "list_dir") < PASS_BAR, \
        "工具选错（L3）不得通过"


def test_score_qa_and_bar():
    assert score_qa("GRPO 与 SFT", ["GRPO"]) == 1.0
    assert PASS_BAR == 0.7


# ── 回归：路由覆盖扩容后的评测任务 ──────────────────────────────
def test_route_still_covers_expanded_evalset():
    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    hit = sum(1 for t in es["action_tasks"] if is_task(t["prompt"]))
    assert hit >= len(es["action_tasks"]) - 1


@pytest.mark.skipif(not GRPO5_OK, reason="grpo5 checkpoint 未生成")
def test_grpo5_artifacts():
    m = json.loads((ART / "grpo5_metrics.json").read_text(encoding="utf-8"))
    assert m["algo"] == "GRPO+PRM+AnchorSFT"
    assert "alpha_sft" in m and m["final_reward"] >= 0
    h = m["history"][-1]
    assert "anchor_nll" in h, "对齐税锚定损失必须入档"
