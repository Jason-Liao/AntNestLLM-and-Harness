# -*- coding: utf-8 -*-
"""Agent 8（测试开发工程师）：M7 质量门禁 —— 多步穿透 / 参数级对比 / α 调参 / 批量轨迹 / 语料 v6。"""
import json
from pathlib import Path

import pytest
import torch

from antnest_llm.grpo import (build_multistep_pool, build_param_pairs,
                              build_contrast_pairs, TASK_POOL, PARAM_NEG_ARGS,
                              TOOL_DEMO_ARGS, TOOL_ARGS, LEGAL_TOOLS,
                              contrastive_loss, evaluate_response, alpha_at)
from antnest_llm.sft import MULTITURN_SPECS, multiturn_ctx, build_multiturn_examples
from antnest_llm.corpus import load_corpus_v6
from antnest_harness.chat_crew import read_batch_tasks, TASK_HINTS
from antnest_harness.tools import ToolRegistry

ART = Path("/workspace/antnest/artifacts")
SFT9_OK = (ART / "sft9_metrics.json").exists()
GRPO9_OK = (ART / "grpo9_metrics.json").exists()


def _fence(act: dict) -> str:
    return "```action\n" + json.dumps(act, ensure_ascii=False) + "\n```"


# ── M7-① 多步穿透池：链式任务逐步入 GRPO ─────────────────────
def test_multistep_pool_expanded():
    pool = build_multistep_pool()
    assert len(pool) == sum(len(t) for _, t in MULTITURN_SPECS)
    assert len(pool) >= 21, "10 条链展开至少 21 个训练项"


def test_multistep_pool_ctx_format_matches_eval():
    """历史回填格式必须与 sft/eval 逐字一致（第N步已完成，使用了 X）。"""
    pool = build_multistep_pool()
    prompt, (kind, tool) = pool[0]
    assert "已完成，使用了" in prompt and "继续下一步" in prompt
    assert kind in ("tool", "finish")
    assert tool is None or tool in LEGAL_TOOLS
    # 与 sft 的三元组上下文同构
    task, trans = MULTITURN_SPECS[0]
    assert prompt == multiturn_ctx(task, trans[0][0], trans[0][1])


def test_multistep_pool_finish_steps_present():
    """链尾 finish 步也须入池（教模型'何时收工'）。"""
    kinds = {k for _, (k, _) in build_multistep_pool()}
    assert kinds == {"tool", "finish"}


def test_multistep_eval_separation():
    """多步池任务措辞不得与评测集 multi_tasks 重合（防过拟合）。"""
    es = json.loads((Path("/workspace/antnest/evals/evalset.json")
                     ).read_text(encoding="utf-8"))
    eval_prompts = {t["prompt"] for t in es["multi_tasks"]}
    for prompt, _ in build_multistep_pool():
        # 评测任务是完整任务句；训练池 prompt = 任务句 + 历史回填
        for ep in eval_prompts:
            assert not prompt.startswith(ep), "训练多步任务与评测集措辞泄漏"


# ── M7-② 参数级对比学习（L4 攻坚）────────────────────────────
def test_param_neg_args_are_wrong_keys():
    """每个参数级负例的键必须是错误键（与 TOOL_ARGS 不交）。"""
    for tool, neg in PARAM_NEG_ARGS.items():
        assert set(neg) & TOOL_ARGS[tool] == set(), \
            f"{tool} 负例不应含合法键（{set(neg) & TOOL_ARGS[tool]}）"


def test_param_pairs_same_tool_diff_keys():
    """参数对：正/负例同工具、同格式、仅参数键不同。"""
    pairs = build_param_pairs(TASK_POOL + build_multistep_pool())
    assert len(pairs) >= 35, "主循环用的训练池应产出 35+ 参数对"
    for prompt, pos, neg in pairs[:8]:
        assert '"action": "tool"' in pos and '"action": "tool"' in neg
        # 工具名一致（提取 name 字段）
        pn = json.loads(pos.split("```action")[1].strip("` \n"))["name"]
        nn = json.loads(neg.split("```action")[1].strip("` \n"))["name"]
        assert pn == nn, "参数级对比的正负例必须是同一工具"


def test_param_contrast_gives_gradient():
    """参数级对比损失可运行且有限（含梯度通路）。"""
    from antnest_llm.model import TinyGPT
    from antnest_llm.tokenizer import CharTokenizer
    tok = CharTokenizer("参数对比锚定检查")
    m = TinyGPT(len(tok), 16, 2, 1, 32)
    pairs = build_param_pairs(TASK_POOL)[:2]
    loss = contrastive_loss(m, tok, pairs)
    assert loss is not None and torch.isfinite(loss)


def test_prm_l4_param_keys_still_scored():
    """PRM L4：参数键错误（同工具）应丢 L4 分。"""
    exp = ("tool", "grep")
    full = _fence({"action": "tool", "name": "grep",
                   "args": {"p": "/workspace/README.md", "q": "Harness"}})
    wrong = _fence({"action": "tool", "name": "grep",
                    "args": {"path": "/workspace/README.md", "query": "Harness"}})
    assert evaluate_response(full, exp) > evaluate_response(wrong, exp)


# ── M7-③ α 调参实验（对齐税根治方向）─────────────────────────
def test_alpha_schedule_bounds():
    """α 调度器上界可到 0.7（M7-3 重锚定实验档位）。"""
    assert alpha_at(1, 40, 0.2, 0.7) == 0.7
    assert alpha_at(40, 40, 0.2, 0.7) == 0.2
    assert alpha_at(20, 40, 0.2, 0.7) == pytest.approx(0.7 - 0.5 * 19 / 39)


@pytest.mark.skipif(not GRPO9_OK, reason="grpo9 未训练")
def test_grpo9_alpha_experiment_logged():
    """M7-3 实验：grpo9 使用 α_max=0.7 / α_min=0.2 并留档。"""
    m = json.loads((ART / "grpo9_metrics.json").read_text(encoding="utf-8"))
    assert m["alpha_sft"] == 0.7 and m["alpha_min"] == 0.2
    assert "MultistepPool" in m["algo"] and "ParamContrast" in m["algo"]


# ── M7-④ 批量轨迹积累 ────────────────────────────────────────
def test_read_batch_tasks(tmp_path):
    f = tmp_path / "tasks.txt"
    f.write_text("# 注释行\n\n统计交付物数量\n列出 artifacts 目录\n", encoding="utf-8")
    msgs = read_batch_tasks(str(f))
    assert msgs == ["统计交付物数量", "列出 artifacts 目录"]


def test_batch_tasks_are_routed_to_crew():
    """批量任务文件的典型消息必须命中任务路由（否则轨迹无法积累）。"""
    for msg in ("统计交付物数量", "列出 artifacts 目录", "检索 README 里的关键词",
                "找出所有 py 文件", "把结论写成报告"):
        assert any(h in msg for h in TASK_HINTS), f"批量任务 {msg!r} 未命中路由"


def test_traj_grew_after_batch():
    """M7-4 批量模式后轨迹应显著增长（3 条 → 8+ 条）。"""
    if not TRAJ.exists():
        pytest.skip("trajs.jsonl 不存在")
    n = sum(1 for line in TRAJ.read_text(encoding="utf-8").splitlines()
            if line.strip())
    assert n >= 8, f"批量积累后轨迹应有 8+ 条，实际 {n}"


TRAJ = ART / "trajs.jsonl"


# ── M7-⑤ 语料 v6（运行数据再进一层）──────────────────────────
def test_corpus_v6_contains_runtime_data():
    """v6 在 v5 之上纳入 eval_compare / chat_log，且不含评测集。"""
    text = load_corpus_v6()
    assert "《README》" in text
    assert "《eval_compare.json》" in text or "《chat_log.md》" in text
    es = json.loads((Path("/workspace/antnest/evals/evalset.json")
                     ).read_text(encoding="utf-8"))
    for t in es["action_tasks"][:10]:
        assert t["prompt"] not in text, f"评测措辞泄漏入语料：{t['prompt']}"


def test_corpus_v6_superset_of_v5():
    from antnest_llm.corpus import load_corpus_v5
    assert load_corpus_v5() in load_corpus_v6()


# ── 训练产物存在性（M7 全链路）────────────────────────────────
@pytest.mark.skipif(not SFT9_OK, reason="sft9 未训练")
def test_sft9_artifacts():
    m = json.loads((ART / "sft9_metrics.json").read_text(encoding="utf-8"))
    assert m["best_val_loss"] < 2.0, "sft9 验证损失应收敛到 2 以下"
    assert m["n_train"] > 100, "训练样本应含动作/多轮/轨迹回流"


@pytest.mark.skipif(not GRPO9_OK, reason="grpo9 未训练")
def test_grpo9_reward_history():
    m = json.loads((ART / "grpo9_metrics.json").read_text(encoding="utf-8"))
    assert m["best_reward"] > 0.3, "PRM 通过率过低，多步池可能未生效"
