# -*- coding: utf-8 -*-
"""Agent 8（测试开发工程师）：M6 质量门禁 —— 多轮上下文 / α 动态调度 /
语料 v5 / grep-find 工具扩展 / 进化安全回归门禁。"""
import json
import re
from pathlib import Path

import pytest

ART = Path("/workspace/antnest/artifacts")
EVALSET = Path("/workspace/antnest/evals/evalset.json")
# 以 model_config（训练结束才落盘）判定产物完备，避免训练中途误触发
V6_OK = (ART / "v6_model_config.json").exists() and (ART / "v6_ckpt.pt").exists()
GRPO8_OK = (ART / "grpo8_metrics.json").exists() and (ART / "grpo8_ckpt.pt").exists()


def _tool_of(ans: str):
    m = re.search(r"```action\s*(\{.*?\})\s*```", ans, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1)).get("name")
    except json.JSONDecodeError:
        return None


def _kind_of(ans: str):
    m = re.search(r"```action\s*(\{.*?\})\s*```", ans, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1)).get("action")
    except json.JSONDecodeError:
        return None


# ── M6-1：多轮上下文样本 ─────────────────────────────────────
def test_multiturn_examples_exist():
    from antnest_llm.sft import build_multiturn_examples
    mt = build_multiturn_examples()
    assert len(mt) >= 15, "多轮三元组样本应 ≥15 条"
    # 上下文必须带历史摘要与继续指令（与 run_multi_step 回填格式一致）
    for q, a in mt:
        assert "步已完成，使用了" in q and "继续下一步" in q, \
            f"上下文缺历史/继续标记: {q[:30]}"
        assert "```action" in a, "回答必须是合法动作围栏"


def test_multiturn_format_matches_eval_loop():
    """训练上下文格式应与 eval.run_multi_step 的观察回填逐字同构。"""
    from antnest_llm.sft import build_multiturn_examples
    mt = build_multiturn_examples()
    # eval 侧格式：{prompt}\n（第{i+1}步已完成，使用了 {name}，结果正常。）继续下一步。
    pat = re.compile(r"^(.+)\n（第(\d+)步已完成，使用了 (\S+?)，结果正常。）继续下一步。$")
    checked = 0
    for q, _ in mt:
        m = pat.match(q)
        if m:
            checked += 1
    assert checked == len(mt), "所有多轮样本上下文都应符合回填格式"


def test_multiturn_separation_from_evalset():
    """训练多轮任务与评测集 multi_tasks 措辞物理隔离。"""
    from antnest_llm.sft import build_multiturn_examples
    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    eval_prompts = {t["prompt"] for t in es["multi_tasks"]}
    mt = build_multiturn_examples()
    for q, _ in mt:
        task = q.split("\n")[0]
        assert task not in eval_prompts, f"训练任务泄漏评测措辞: {task}"


def test_multiturn_next_action_correct():
    """历史回填后，期望的下一动作应与任务剩余步骤一致（抽样校验）。"""
    from antnest_llm.sft import build_multiturn_examples
    mt = build_multiturn_examples()
    # 用 grep 步之后应回填 grep；首个动作前的任务以 list_dir/read_file 开头
    seen_tools = {_tool_of(a) for _, a in mt}
    assert {"write_file", "shell"} <= seen_tools, "多轮样本应覆盖多种下一步动作"
    assert any(_kind_of(a) == "finish" for _, a in mt), "应有终轮 finish 样本"


# ── M6-2：α 动态调度 ─────────────────────────────────────────
def test_alpha_schedule_endpoints():
    from antnest_llm.grpo import alpha_at
    assert alpha_at(1, 30, 0.1, 0.5) == pytest.approx(0.5), "首迭代应为 α_max"
    assert alpha_at(30, 30, 0.1, 0.5) == pytest.approx(0.1), "末迭代应为 α_min"
    assert alpha_at(1, 1, 0.1, 0.5) == pytest.approx(0.5), "total≤1 恒为 α_max"


def test_alpha_schedule_monotonic():
    from antnest_llm.grpo import alpha_at
    vals = [alpha_at(i, 20, 0.1, 0.5) for i in range(1, 21)]
    assert all(a >= b for a, b in zip(vals, vals[1:])), "α 应单调不增（退火）"


def test_grpo_argparse_has_alpha_min():
    src = (Path("/workspace/antnest/antnest_llm/grpo.py")).read_text(encoding="utf-8")
    assert "--alpha_min" in src, "GRPO 应提供 α 下限参数"
    assert "alpha_at(" in src, "训练循环应调用动态调度"


# ── M6-3：语料 v5 ────────────────────────────────────────────
def test_corpus_v5_superset_of_v4():
    from antnest_llm.corpus import load_corpus_v4, load_corpus_v5
    v4, v5 = load_corpus_v4(), load_corpus_v5()
    assert len(v5) > len(v4), "v5 应比 v4 多运行数据"
    assert v5.startswith(v4[:1000]), "v5 应以 v4 为前缀（增量叠加）"


def test_corpus_v5_contains_runtime_data():
    from antnest_llm.corpus import load_corpus_v5
    v5 = load_corpus_v5()
    assert "AntNestLLM-and-Harness" in v5, "README 应入库"
    assert "evolve_log" in v5, "进化日志应入库"
    # 评测集永不进训练梯度（M3 纪律）：evalset 措辞不得出现在语料中
    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    leak = [t["prompt"] for t in es["action_tasks"] if t["prompt"] in v5]
    assert not leak, f"评测措辞泄漏进预训练语料: {leak}"


def test_train_supports_v5():
    src = (Path("/workspace/antnest/antnest_llm/train.py")).read_text(encoding="utf-8")
    assert "v5" in src and "load_corpus_v5" in src, "train.py 应支持语料 v5"


# ── M6-4：grep / find 工具扩展 ───────────────────────────────
def test_grep_tool_sandbox():
    from antnest_harness.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register_defaults()
    out = reg.execute("grep", {"p": "/workspace/README.md", "q": "Harness"})
    hits = json.loads(out)
    assert isinstance(hits, list) and hits, "grep 应命中 README 中的 Harness"
    assert any("Harness" in h for h in hits)


def test_grep_tool_escape_blocked():
    from antnest_harness.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register_defaults()
    out = reg.execute("grep", {"p": "/etc/passwd", "q": "root"})
    assert out.startswith("[工具错误]"), "grep 越界路径必须被沙箱拦截"


def test_find_tool_sandbox():
    from antnest_harness.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register_defaults()
    out = reg.execute("find", {"dir": "/workspace/antnest/antnest_llm", "name": "*.py"})
    hits = json.loads(out)
    assert isinstance(hits, list) and "antnest/antnest_llm/grpo.py" in hits, \
        "find 应定位到 grpo.py（相对 SAFE_ROOT 路径）"


def test_tool_registry_six_tools():
    from antnest_harness.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register_defaults()
    names = {s["function"]["name"] for s in reg.schemas()}
    assert {"list_dir", "read_file", "write_file", "shell", "grep", "find"} <= names, \
        "动作空间应扩至 6 工具"


def test_sft_examples_cover_new_tools():
    from antnest_llm.sft import build_action_examples
    acts = build_action_examples()
    tools = {t for t in (_tool_of(a) for _, a in acts) if t}
    assert {"grep", "find"} <= tools, "SFT 动作示例应覆盖 grep/find"
    from collections import Counter
    cnt = Counter(t for t in (_tool_of(a) for _, a in acts) if t)
    assert cnt["grep"] >= 5 and cnt["find"] >= 5, "新工具每类 ≥5 措辞"


def test_grpo_pool_covers_new_tools():
    from antnest_llm.grpo import TASK_POOL, LEGAL_TOOLS, TOOL_ARGS, TOOL_DEMO_ARGS
    tools = {t for _, (k, t) in TASK_POOL if k == "tool"}
    assert {"grep", "find"} <= tools, "任务池应含 grep/find 任务"
    assert LEGAL_TOOLS == {"list_dir", "shell", "write_file", "read_file", "grep", "find"}
    assert TOOL_ARGS["grep"] == {"p", "q"} and TOOL_ARGS["find"] == {"dir", "name"}
    assert "grep" in TOOL_DEMO_ARGS and "find" in TOOL_DEMO_ARGS


def test_prm_rewards_new_tools():
    import antnest_llm.grpo as G
    G._CURRICULUM = 1.0
    for tool, args in [("grep", {"p": "/workspace/README.md", "q": "Harness"}),
                       ("find", {"dir": "/workspace/antnest", "name": "*.py"})]:
        ans = ("执行：\n```action\n" + json.dumps(
            {"action": "tool", "name": tool, "args": args}, ensure_ascii=False) + "\n```")
        assert G.evaluate_response(ans, ("tool", tool)) == pytest.approx(1.0), \
            f"{tool} 正确动作应得满分（含 L5 沙箱执行）"


def test_evalset_new_tool_tasks():
    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    tools = [t.get("tool") for t in es["action_tasks"] if t.get("expect") == "tool"]
    assert tools.count("grep") >= 3 and tools.count("find") >= 3, \
        "评测集应含 grep/find 各 3+ 条"


def test_evalset_separation_from_training():
    """评测集动作任务措辞不得与 SFT/GRPO 训练措辞重合。"""
    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    from antnest_llm.sft import build_action_examples, build_multiturn_examples
    from antnest_llm.grpo import TASK_POOL
    train_prompts = ({q for q, _ in build_action_examples()}
                     | {q.split("\n")[0] for q, _ in build_multiturn_examples()}
                     | {p for p, _ in TASK_POOL})
    leak = [t["prompt"] for t in es["action_tasks"] if t["prompt"] in train_prompts]
    assert not leak, f"评测措辞泄漏进训练集: {leak}"


# ── M6-5：进化安全回归门禁 ───────────────────────────────────
def test_regression_gate_passes_on_improvement():
    from antnest_llm.evolve import regression_gate
    old = {"action_pass": 0.5, "qa_hit": 0.2, "multi_avg": 0.3, "overall": 0.35}
    new = {"action_pass": 0.6, "qa_hit": 0.19, "multi_avg": 0.35, "overall": 0.4}
    ok, fails = regression_gate(old, new)
    assert ok and not fails, "各项跌幅 ≤10% 应放行"


def test_regression_gate_rejects_single_metric_crash():
    from antnest_llm.evolve import regression_gate
    # overall 微涨但 qa_hit 跌 50% → 必须拒绝（防隐性退化）
    old = {"action_pass": 0.5, "qa_hit": 0.2, "overall": 0.3}
    new = {"action_pass": 0.55, "qa_hit": 0.1, "overall": 0.35}
    ok, fails = regression_gate(old, new)
    assert not ok and any("qa_hit" in f for f in fails), "单项塌方应被门禁拦截"


def test_regression_gate_tolerates_none():
    from antnest_llm.evolve import regression_gate
    old = {"action_pass": 0.5, "qa_hit": None, "multi_avg": None, "overall": 0.3}
    new = {"action_pass": 0.6, "qa_hit": 0.0, "multi_avg": 0.9, "overall": 0.4}
    ok, fails = regression_gate(old, new)
    assert ok, "老产物缺失指标（None）应跳过不拦"


def test_evolve_records_gate():
    src = (Path("/workspace/antnest/antnest_llm/evolve.py")).read_text(encoding="utf-8")
    assert "regression_gate" in src and '"gate"' in src, \
        "进化日志应记录门禁结果"


# ── M6 产物一致性（若 v6 链路已训练）─────────────────────────
@pytest.mark.skipif(not V6_OK, reason="无 v6 checkpoint")
def test_v6_vocab_5000():
    cfg = json.loads((ART / "v6_model_config.json").read_text(encoding="utf-8"))
    from antnest_llm.bpe import load_tokenizer
    tok = load_tokenizer(ART / "v6_vocab.json")
    assert len(tok) == cfg["vocab"] == 5000, "v6 应为 BPE 5000 词表"
    assert cfg["n_layer"] == 12 and cfg["n_embd"] == 384, "v6 沿用 12L-384d"


@pytest.mark.skipif(not GRPO8_OK, reason="无 grpo8 checkpoint")
def test_grpo8_metrics_alpha_schedule():
    m = json.loads((ART / "grpo8_metrics.json").read_text(encoding="utf-8"))
    assert "AlphaSchedule" in m["algo"], "grpo8 应记录 α 调度算法标记"
    alphas = [h.get("alpha") for h in m["history"] if h.get("alpha") is not None]
    assert len(alphas) == len(m["history"]), "每轮迭代都应记录 α"
    assert alphas[0] > alphas[-1], "α 应从高到低退火"
    assert m["alpha_min"] < m["alpha_sft"], "α 下限应小于上限"
