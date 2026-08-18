# -*- coding: utf-8 -*-
"""Agent 8（测试开发工程师）：M3 质量门禁 —— GRPO / 评测分离 / ChatCrew / 语料 v4。"""
import json
from pathlib import Path

import pytest

from antnest_llm.grpo import evaluate_response, TASK_POOL
from antnest_llm.eval import score_action, score_qa, EVALSET
from antnest_llm.corpus import load_corpus_v4
from antnest_harness.chat_crew import is_task, TASK_HINTS
from antnest_harness.llm import MockLLM
from antnest_harness.crew import NestCrew
from antnest_harness.tools import ToolRegistry

ART = Path("/workspace/antnest/artifacts")
GRPO_OK = (ART / "grpo_ckpt.pt").exists()


# ── GRPO 奖励函数 ───────────────────────────────────────────────
def test_reward_full_score():
    text = '好的：\n```action\n{"action":"tool","name":"list_dir","args":{"p":"/tmp"}}\n```'
    assert evaluate_response(text, "tool") == 1.0


def test_reward_partial():
    fence_only = '```action\n{坏json}\n```'
    assert 0.0 < evaluate_response(fence_only, "tool") <= 0.5
    assert evaluate_response("我不明白。", "tool") == 0.0  # 无格式要素


def test_reward_shaping():
    # shaping：具备格式要素但无完整围栏 → 非零部分分
    s = evaluate_response('输出 "action":"tool" 这样', "tool")
    assert 0.0 < s < 0.3


def test_reward_type_mismatch():
    text = '```action\n{"action":"tool","name":"shell","args":{"cmd":"ls"}}\n```'
    assert evaluate_response(text, "finish") < 1.0


def test_task_pool_integrity():
    assert len(TASK_POOL) >= 8
    assert all(e in ("tool", "finish") for _, e in TASK_POOL)


# ── 评测与训练分离 ─────────────────────────────────────────────
def test_evalset_exists_and_disjoint():
    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    assert len(es["action_tasks"]) >= 5 and len(es["qa_tasks"]) >= 3
    train_prompts = {p for p, _ in TASK_POOL}
    eval_prompts = {t["prompt"] for t in es["action_tasks"]}
    assert not (train_prompts & eval_prompts), "评测任务不得与训练任务同措辞"


def test_score_action_strict():
    assert score_action("随便说说", "tool") == 0.0
    t = '```action\n{"action":"finish","result":"完成"}\n```'
    assert score_action(t, "finish") == 1.0


def test_score_qa_hit():
    assert score_qa("antNest LLM 与 Harness", ["antNest LLM", "Harness"]) == 1.0
    assert score_qa("不知道", ["BPE"]) == 0.0


# ── ChatCrew 路由 ──────────────────────────────────────────────
def test_route_task_vs_chat():
    assert is_task("帮我列出目录文件")
    assert is_task("统计一下交付物数量")
    assert not is_task("你好呀")
    assert not is_task("蚁巢计划是什么")


def test_route_covers_evalset_tasks():
    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    hit = sum(1 for t in es["action_tasks"] if is_task(t["prompt"]))
    assert hit >= len(es["action_tasks"]) - 1, "评测任务应几乎全部路由到 Crew"


def test_crew_with_mock_llm_loop():
    # MockLLM 脚本化驱动三角色闭环（planner→builder→reviewer）
    script = [
        '```action\n{"action":"finish","result":"计划：列目录并统计"}\n```',
        '```action\n{"action":"tool","name":"list_dir","args":{"p":"/workspace/antnest_team/outputs"}}\n```',
        '```action\n{"action":"finish","result":"已列出并统计完成"}\n```',
        '```action\n{"action":"finish","result":"审查通过，目标达成"}\n```',
    ]
    tools = ToolRegistry()
    tools.register_defaults()
    crew = NestCrew(llm=MockLLM(script), tools=tools)
    out = crew.kickoff("列出交付物目录")
    assert out["plan"] and out["review"]
    assert "审查通过" in out["review"]


# ── 语料 v4 ────────────────────────────────────────────────────
def test_corpus_v4_external():
    text = load_corpus_v4()
    assert "学而时习之" in text, "外部公有领域语料应并入 v4"
    assert "PYTHON SOFTWARE FOUNDATION" in text, "PSF 许可文本应并入 v4"
    assert len(text) > 100000


@pytest.mark.skipif(not GRPO_OK, reason="GRPO checkpoint 未生成")
def test_grpo_artifacts_consistent():
    m = json.loads((ART / "grpo_metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((ART / "grpo_model_config.json").read_text(encoding="utf-8"))
    vocab = json.loads((ART / "grpo_vocab.json").read_text(encoding="utf-8"))
    assert m["algo"] == "GRPO" and m["final_reward"] >= 0
    assert len(vocab["base"]) + len(vocab["merges"]) == cfg["vocab"]
