# -*- coding: utf-8 -*-
"""Agent 18/19 + Agent 25：antNest LLM RL 环境（GRPO，M6 版）。

M3 基线：GRPO（组相对策略优化）——组采样归一化优势 + ratio 裁剪 + KL 锚定。
M4 升级两项：
  ① 过程奖励 PRM（Agent 18/19）：奖励细分五级——
       L1 格式围栏(0.2) → L2 动作类型(0.2) → L3 工具选择正确(0.3)
       → L4 参数键完备(0.15) → L5 沙箱真实执行成功(0.15)
     其中 L5 在 DSec 沙箱内真实执行候选动作，以"结果可用"给分（真过程奖励）。
  ② 对齐税优化（Agent 25，REINFORCE++ 风格）：每次迭代混入 SFT 锚定批次，
     total = GRPO 损失 + α·NLL(锚定样本)，缓解 RL 后 QA 能力遗忘。
M6 升级两项：
  ③ α 动态调度（M6-2 对齐税再平衡）：锚定系数从 α_max 线性退火至 α_min——
     前期重锚定保 QA 不遗忘，后期释放策略优化空间。
  ④ 动作空间 4 → 6（M6-4）：新增 grep（内容检索）/ find（名称查找）
     参数化任务，TASK_POOL 同步扩容。

用法：python -m antnest_llm.grpo --iters 40 --base_prefix sft4 --out_prefix grpo5
产出：artifacts/{grpo5_ckpt.pt, grpo5_metrics.json, grpo5_model_config.json,
                grpo5_vocab.json}
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .bpe import load_tokenizer
from .model import TinyGPT
from .sft import U, A, greedy_answer, build_examples, encode_example

ART = Path(__file__).resolve().parent.parent / "artifacts"

LEGAL_TOOLS = {"list_dir", "shell", "write_file", "read_file", "grep", "find"}
# 参数键完备性：每个工具必需的 args 键（L4）
TOOL_ARGS = {"list_dir": {"p"}, "read_file": {"p"},
             "write_file": {"p", "c"}, "shell": {"cmd"},
             "grep": {"p", "q"}, "find": {"dir", "name"}}

# 任务池：prompt → (期望动作类型, 期望工具名)
# expect_tool 为 None 表示 finish 类任务
# M5-1 扩容：10 → 22（与 SFT 同分布增补措辞，覆盖 4 工具 × 近义表达）
TASK_POOL = [
    ("请列出 extracted 目录的文件。", ("tool", "list_dir")),
    ("查看 /workspace 下的交付物目录。", ("tool", "list_dir")),
    ("我想知道 outputs 目录下有哪些交付物。", ("tool", "list_dir")),
    ("帮我查看 antnest 项目的目录结构。", ("tool", "list_dir")),
    ("列一下 artifacts 里都存了什么。", ("tool", "list_dir")),
    ("统计团队交付物数量。", ("tool", "shell")),
    ("数一数 outputs 有多少个文件。", ("tool", "shell")),
    ("帮我查一下词表有多少行。", ("tool", "shell")),
    ("搜索文件里的关键词 antNest。", ("tool", "shell")),
    ("统计 README 的字数。", ("tool", "shell")),
    ("把结论写成报告。", ("tool", "write_file")),
    ("把这些内容保存成报告文件。", ("tool", "write_file")),
    ("帮我保存这些笔记。", ("tool", "write_file")),
    ("生成一份总结文档。", ("tool", "write_file")),
    ("读取报告内容。", ("tool", "read_file")),
    ("看看报告里写了什么。", ("tool", "read_file")),
    ("我想看 README 的内容。", ("tool", "read_file")),
    ("查看模型配置。", ("tool", "read_file")),
    # M6-4：grep / find 参数化检索任务（动作空间 4 → 6）
    ("帮我找找 README 里包含 Harness 的行。", ("tool", "grep")),
    ("在词表里搜一下这个词条。", ("tool", "grep")),
    ("报告里哪里提到了沙箱？帮我找出来。", ("tool", "grep")),
    ("在源码文件里查找一下 sandbox。", ("tool", "grep")),
    ("找出仓库里所有 python 文件。", ("tool", "find")),
    ("查一下 antnest 下的 md 文件有哪些。", ("tool", "find")),
    ("帮我找出所有 json 配置文件。", ("tool", "find")),
    ("找一下 tests 目录里的测试文件。", ("tool", "find")),
    ("任务已完成，请结束。", ("finish", None)),
    ("所有工作已结束。", ("finish", None)),
    ("做完了，收工吧。", ("finish", None)),
    ("可以结束了。", ("finish", None)),
]

# 每工具演示参数（对比学习正/负例构造用）
TOOL_DEMO_ARGS = {
    "list_dir": {"p": "/workspace/extracted"},
    "read_file": {"p": "/workspace/antnest/artifacts/report.md"},
    "write_file": {"p": "/workspace/antnest/artifacts/report.md", "c": "antNest 报告"},
    "shell": {"cmd": "ls /workspace/antnest_team/outputs | wc -l"},
    "grep": {"p": "/workspace/README.md", "q": "Harness"},
    "find": {"dir": "/workspace/antnest", "name": "*.py"},
}

# M7-2：参数级负例——同工具、同格式、参数键名错误（L4 直训）。
# 选用"语义正确但键名拼写错误"的负例（如 path/query 替代 p/q），
# 这是 LLM 生成动作时最常见的参数级错误形态。
PARAM_NEG_ARGS = {
    "list_dir": {"path": "/workspace/extracted"},
    "read_file": {"file": "/workspace/antnest/artifacts/report.md"},
    "write_file": {"file": "/workspace/antnest/artifacts/report.md",
                   "content": "antNest 报告"},
    "shell": {"command": "ls /workspace/antnest_team/outputs | wc -l"},
    "grep": {"path": "/workspace/README.md", "query": "Harness"},
    "find": {"directory": "/workspace/antnest", "pattern": "*.py"},
}


# ── M7-1：多步穿透池（PRM 穿透链式任务）────────────────────
def build_multistep_pool() -> list:
    """把多步链展开为逐步 GRPO 训练项：每步 = (带历史的 prompt, 期望动作)。

    SFT 三元组只教了"见历史选对动作"的模仿；GRPO 多步池再让 PRM 对
    每一步逐步打分（组内对比 + L1-L5 过程奖励），链式任务的全通率
    攻坚从"见过分布"升级为"在分布上被强化"。历史回填格式与
    eval.run_multi_step / sft.multiturn_ctx 逐字一致。
    """
    from .sft import MULTITURN_SPECS, multiturn_ctx
    pool = []
    for task, trans in MULTITURN_SPECS:
        for step, used, (kind, tool, _args) in trans:
            pool.append((multiturn_ctx(task, step, used), (kind, tool)))
    return pool


# ── M6-2：α 动态调度（对齐税再平衡）────────────────────────
def alpha_at(it: int, total: int, a_min: float = 0.1, a_max: float = 0.5) -> float:
    """锚定系数 α 的线性退火调度：iter 1 → α_max（重锚定保 QA），
    iter total → α_min（释放策略优化空间）。total≤1 时恒为 α_max。
    """
    if total <= 1:
        return a_max
    t = min(1.0, max(0.0, (it - 1) / (total - 1)))
    return a_max - (a_max - a_min) * t


def _sandbox():
    """惰性构建沙箱工具注册表（PRM L5 真实执行用）。"""
    from antnest_harness.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register_defaults()
    return reg


_SBOX = None
# 奖励课程系数（M4-①b）：早期放宽 L3-L5（先学会格式，再学选对工具），
# 随迭代推进升至 1.0（完全严格 PRM）。由 main() 按迭代更新。
_CURRICULUM = 1.0


def sandbox_ok(action: dict) -> bool:
    """PRM L5：在 DSec 沙箱内执行候选动作，返回结果是否可用。"""
    global _SBOX
    if _SBOX is None:
        _SBOX = _sandbox()
    if action.get("action") != "tool":
        return True  # finish 无需执行
    obs = _SBOX.execute(action.get("name", ""), action.get("args", {}))
    return not (obs.startswith("[工具错误]") or obs.startswith("未知工具"))


def evaluate_response(text: str, expect: tuple) -> float:
    """PRM 五级过程奖励 ∈ [0,1]（expect = (动作类型, 期望工具名)）。

    L1 格式 0.2 | L2 类型 0.2 | L3 工具选择 0.3 | L4 参数键 0.15 | L5 沙箱执行 0.15
    shaping：具备格式要素但无完整围栏 → 最多 0.06（冷启动引导，不参与满分）。
    """
    expect_kind, expect_tool = expect
    m = re.search(r"```action\s*(\{.*?\})\s*```", text, re.S)
    if not m:
        s = 0.0
        if "```action" in text:
            s += 0.02
        if '"action"' in text:
            s += 0.02
        if '"tool"' in text or '"finish"' in text:
            s += 0.02
        return s
    score = 0.2  # L1
    try:
        act = json.loads(m.group(1))
    except json.JSONDecodeError:
        return score
    if act.get("action") not in ("tool", "finish"):
        return score
    if act["action"] != expect_kind:
        return score
    score += 0.2  # L2
    if expect_kind == "finish":
        return score + (0.6 if str(act.get("result", "")).strip() else 0.0)
    # tool 类：L3 工具选择 / L4 参数键 / L5 沙箱执行（受课程系数调节）
    name = act.get("name")
    if name != expect_tool:
        # 课程早期选错工具也给部分分（合法工具即 0.3·c 的一半），
        # 避免严格 L3 在冷启动期把奖励信号切断
        return score + (0.15 * _CURRICULUM if name in LEGAL_TOOLS else 0.0)
    score += 0.3 * _CURRICULUM  # L3：选对工具（过程奖励核心）
    args = act.get("args")
    if isinstance(args, dict) and TOOL_ARGS.get(name, set()) <= set(args):
        score += 0.15 * _CURRICULUM  # L4
    if sandbox_ok(act):
        score += 0.15 * _CURRICULUM  # L5
    return score


@torch.no_grad()
def sample_group(model, tok, prompt: str, g: int, max_new: int, temperature: float):
    """对同一 prompt 采样 g 个响应，返回 (ids 列表, prompt 长度)。"""
    pids = tok.encode(f"{U}{prompt}\n{A}") or [0]
    ctx = model.block_size
    idx = torch.tensor([pids[-ctx:]] * g, dtype=torch.long)
    p_len = idx.size(1)
    for _ in range(max_new):
        logits, _ = model(idx[:, -ctx:])
        probs = F.softmax(logits[:, -1, :] / max(temperature, 1e-5), dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        if idx.size(1) >= ctx:
            break
    return [r.tolist() for r in idx], p_len


def seq_logprob(model, ids: list, p_len: int):
    """响应区间的 token logprob 总和（含梯度）。"""
    s = max(p_len, 1)
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor(ids[s:], dtype=torch.long)
    if y.numel() == 0 or x.size(1) == 0:
        return None, None
    logits, _ = model(x)
    n = min(y.numel(), logits.size(1) - (s - 1))
    if n <= 0:
        return None, None
    nll = F.cross_entropy(logits[0, s - 1: s - 1 + n], y[:n], reduction="none")
    return -nll.sum(), n


def grpo_loss(model, ref, ids: list, p_len: int, adv: float, clip=0.2, beta=0.05):
    """单条响应的 GRPO 目标：-min(rA, clip(r)A) + beta·KL(π‖π_ref)。"""
    lp, n = seq_logprob(model, ids, p_len)
    if lp is None:
        return None
    with torch.no_grad():
        rlp, rn = seq_logprob(ref, ids, p_len)
        if rlp is None:
            return None
    logratio = (lp / n) - (rlp / rn)
    ratio = torch.exp(torch.clamp(logratio, -5.0, 5.0))
    un = -adv * ratio
    cl = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    loss = torch.max(un, cl)
    # 逐 token KL（k3 估计）
    s = max(p_len, 1)
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    logits, _ = model(x)
    y = torch.tensor(ids[s:], dtype=torch.long)
    nn_ = min(y.numel(), logits.size(1) - (s - 1))
    if nn_ <= 0:
        return loss
    pl = F.log_softmax(logits[0, s - 1: s - 1 + nn_], dim=-1)
    with torch.no_grad():
        rlogits, _ = ref(x)
        rl = F.log_softmax(rlogits[0, s - 1: s - 1 + nn_], dim=-1)
    kl = (torch.exp(pl) * (pl - rl)).sum(-1).mean()
    return loss + beta * kl


def sft_anchor_loss(model, tok, anchors: list, block: int):
    """对齐税锚定（M4-①）：对锚定 QA 批计算 masked NLL（含梯度）。"""
    losses = []
    for q, a in anchors:
        e = encode_example(tok, q, a, block)
        if e is None:
            continue
        x, y = e
        if x.numel() < 2:
            continue
        logits, _ = model(x[None, :])
        # encode_example 已返回对齐的 (x=ids[:-1], y=ids[1:])，长度一致
        nll = F.cross_entropy(logits[0], y, ignore_index=-100)
        losses.append(nll)
    if not losses:
        return None
    return torch.stack(losses).mean()


# ── M5-1：动作空间对比学习（工具选择攻坚）─────────────────
def _action_text(tool: str) -> str:
    return ("执行：\n```action\n" + json.dumps(
        {"action": "tool", "name": tool, "args": TOOL_DEMO_ARGS[tool]},
        ensure_ascii=False) + "\n```")


def build_contrast_pairs(task_pool) -> list:
    """构造 (prompt, 正例响应, 负例响应)：负例 = 同格式但选错工具。

    正误工具仅工具名不同（格式/参数键均合法），对比信号聚焦 L3 工具选择。
    负例工具确定性轮转（hash(prompt)），保证每轮覆盖不同混淆方向。
    """
    pairs = []
    for prompt, (kind, tool) in task_pool:
        if kind != "tool" or tool is None:
            continue
        others = sorted(LEGAL_TOOLS - {tool})
        neg = others[hash(prompt) % len(others)]
        pairs.append((prompt, _action_text(tool), _action_text(neg)))
    return pairs


# ── M7-2：参数级对比学习（L4 攻坚）────────────────────────
def _action_text_args(tool: str, args: dict) -> str:
    return ("执行：\n```action\n" + json.dumps(
        {"action": "tool", "name": tool, "args": args},
        ensure_ascii=False) + "\n```")


def build_param_pairs(task_pool) -> list:
    """参数级对比对：正例 = 正确参数键，负例 = 同工具但键名拼写错误。

    M5-1 的对比负例是"选错工具"（L3），模型对齐后组内奖励已能区分；
    M7-2 补上"选对工具但参数键错"的负例（如 grep 的 path/query 替代
    p/q）——L4 在 PRM 中仅占 0.15，组内对比信号弱，pairwise margin
    直训参数键拼写，检索类工具（grep/find 双参数）受益最大。
    """
    pairs = []
    for prompt, (kind, tool) in task_pool:
        if kind != "tool" or tool is None or tool not in PARAM_NEG_ARGS:
            continue
        pairs.append((prompt, _action_text_args(tool, TOOL_DEMO_ARGS[tool]),
                      _action_text_args(tool, PARAM_NEG_ARGS[tool])))
    return pairs


def resp_logprob(model, tok, prompt: str, resp: str):
    """响应区间平均 token logprob（含梯度，长度归一）。

    超出模型 ctx 时保留尾部（响应优先），保证任意模型尺寸下可用。
    """
    ctx = getattr(model, "block_size", 256)
    rids = tok.encode(resp) or [0]
    pids = tok.encode(f"{U}{prompt}\n{A}") or [0]
    ids = pids + rids
    if len(ids) > ctx + 1:                      # 超长：截头部保响应
        ids = ids[-(ctx + 1):]
    p_len = max(len(ids) - len(rids), 1)
    if len(ids) <= p_len:
        return None
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    logits, _ = model(x)
    lp = F.log_softmax(logits[0], dim=-1)          # (T, V)
    tgt = y[p_len - 1:]                             # 响应区间的目标 token
    n = min(lp.size(0), tgt.numel())
    if n <= 0:
        return None
    idx = torch.arange(n, device=lp.device)
    return lp[idx, tgt[:n]].mean()


def contrastive_loss(model, tok, pairs: list, margin: float = 0.5):
    """pairwise margin 损失：max(0, margin − (lp_pos − lp_neg))。

    同 prompt 下拉开正/误工具响应的 logprob 差，直击 L3（GRPO 组内
    若全组选错则优势归零无梯度，对比学习补上这条梯度通路）。
    """
    losses = []
    for prompt, pos, neg in pairs:
        lp_p = resp_logprob(model, tok, prompt, pos)
        lp_n = resp_logprob(model, tok, prompt, neg)
        if lp_p is None or lp_n is None:
            continue
        losses.append(F.relu(margin - (lp_p - lp_n)))
    if not losses:
        return None
    return torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--group", type=int, default=6, help="组大小 G")
    ap.add_argument("--prompts_per_iter", type=int, default=4)
    ap.add_argument("--anchor_per_iter", type=int, default=3, help="每次迭代锚定样本数")
    ap.add_argument("--alpha_sft", type=float, default=0.5,
                    help="对齐税系数 α 上限（M6-2 动态调度起点）")
    ap.add_argument("--alpha_min", type=float, default=0.1,
                    help="对齐税系数 α 下限（调度终点）")
    ap.add_argument("--lambda_ctr", type=float, default=0.3,
                    help="M5-1 动作空间对比学习系数 λ")
    ap.add_argument("--ctr_per_iter", type=int, default=3,
                    help="每次迭代对比对数")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--base_prefix", default="sft3", help="起点（SFT 产物）")
    ap.add_argument("--out_prefix", default="grpo")
    args = ap.parse_args()

    torch.manual_seed(11)
    bp, op = args.base_prefix, args.out_prefix
    cfg = json.loads((ART / f"{bp}_model_config.json").read_text(encoding="utf-8"))
    tok = load_tokenizer(ART / f"{bp}_vocab.json")
    assert len(tok) == cfg["vocab"], "词表一致性校验失败"

    model = TinyGPT(cfg["vocab"], cfg["n_embd"], cfg["n_head"],
                    cfg["n_layer"], cfg["block_size"])
    model.load_state_dict(torch.load(ART / f"{bp}_ckpt.pt", weights_only=True))
    ref = TinyGPT(cfg["vocab"], cfg["n_embd"], cfg["n_head"],
                  cfg["n_layer"], cfg["block_size"])
    ref.load_state_dict(model.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 对齐税锚定池：复用 SFT QA 数据（知识问答 + 职位问答），不重复采样动作模板
    anchor_pool = [(q, a) for q, a in build_examples()]
    # M7-1 训练池：单步任务 + 多步链逐步展开（PRM 穿透链式任务）
    train_pool = TASK_POOL + build_multistep_pool()
    # 对比池（M5-1 工具级 L3 + M7-2 参数级 L4）
    contrast_pool = build_contrast_pairs(train_pool) + build_param_pairs(train_pool)
    print(f"训练池：{len(train_pool)} 项（含多步 {len(train_pool) - len(TASK_POOL)} 项）")
    print(f"对比池：{len(contrast_pool)} 对（工具级 {len(build_contrast_pairs(train_pool))}"
          f" + 参数级 {len(build_param_pairs(train_pool))}）")
    rng = torch.Generator().manual_seed(3)
    history, t0, best = [], time.time(), -1.0

    for it in range(1, args.iters + 1):
        # 奖励课程：前 warm 阶段 L3-L5 打 2 折起步，10 迭代内线性升至全严格
        globals()["_CURRICULUM"] = 0.2 + 0.8 * min(1.0, (it - 1) / 10)
        sel = [train_pool[i] for i in torch.randperm(len(train_pool), generator=rng)
               [: args.prompts_per_iter]]
        tot_loss, tot_rew, tot_anchor, tot_ctr, n_g = 0.0, 0.0, 0.0, 0.0, 0
        # M6-2：α 动态调度（线性退火：前期重锚定保 QA，后期释放策略空间）
        alpha_t = alpha_at(it, args.iters, args.alpha_min, args.alpha_sft)
        for prompt, expect in sel:
            model.eval()
            group, p_len = sample_group(model, tok, prompt, args.group,
                                        args.max_new, args.temperature)
            rewards = [evaluate_response(tok.decode(g[p_len:]), expect) for g in group]
            r_t = torch.tensor(rewards)
            adv = ((r_t - r_t.mean()) / (r_t.std(unbiased=False) + 1e-4)).tolist()
            tot_rew += r_t.mean().item(); n_g += 1
            model.train()
            # 对齐税锚定批次（与 GRPO 同一 step，联合反传）
            ai = torch.randperm(len(anchor_pool), generator=rng)[: args.anchor_per_iter]
            anchors = [anchor_pool[i] for i in ai]
            g_losses = [l for l in (grpo_loss(model, ref, g_ids, p_len, a)
                                    for g_ids, a in zip(group, adv)) if l is not None]
            if not g_losses:
                continue
            loss_t = torch.stack(g_losses).mean()
            a_nll = sft_anchor_loss(model, tok, anchors, cfg["block_size"])
            if a_nll is not None:
                loss_t = loss_t + alpha_t * a_nll
                tot_anchor += a_nll.item()
            # M5-1 动作空间对比批次（同 step 联合反传）
            ci = torch.randperm(len(contrast_pool), generator=rng)[: args.ctr_per_iter]
            c_loss = contrastive_loss(model, tok, [contrast_pool[i] for i in ci])
            if c_loss is not None:
                loss_t = loss_t + args.lambda_ctr * c_loss
                tot_ctr += c_loss.item()
            opt.zero_grad(set_to_none=True)
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            tot_loss += loss_t.item()
        mean_rew = tot_rew / max(1, n_g)
        mean_anchor = tot_anchor / max(1, n_g)
        mean_ctr = tot_ctr / max(1, n_g)
        history.append({"iter": it, "loss": round(tot_loss, 4),
                        "reward": round(mean_rew, 4),
                        "anchor_nll": round(mean_anchor, 4),
                        "ctr_loss": round(mean_ctr, 4),
                        "alpha": round(alpha_t, 4)})
        if it % 5 == 0 or it == 1:
            print(f"iter {it:>3} | loss {tot_loss:>7.3f} | PRM通过率 {mean_rew:.3f} | "
                  f"锚定NLL {mean_anchor:.3f} | 对比loss {mean_ctr:.3f} | α {alpha_t:.3f}")
        if mean_rew >= best:
            best = mean_rew
            torch.save(model.state_dict(), ART / f"{op}_ckpt.pt")

    tok.save(ART / f"{op}_vocab.json")
    (ART / f"{op}_model_config.json").write_text(
        (ART / f"{bp}_model_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    (ART / f"{op}_metrics.json").write_text(json.dumps(
        {"algo": "GRPO+PRM+AnchorSFT+ContrastiveTools+AlphaSchedule+MultistepPool+ParamContrast", "iters": args.iters,
         "group": args.group, "alpha_sft": args.alpha_sft, "alpha_min": args.alpha_min,
         "lambda_ctr": args.lambda_ctr, "best_reward": round(best, 4),
         "final_reward": history[-1]["reward"],
         "final_anchor_nll": history[-1]["anchor_nll"],
         "seconds": round(time.time() - t0, 1), "history": history},
        ensure_ascii=False, indent=1), encoding="utf-8")
    model.eval()
    demo = greedy_answer(model, tok, "请列出 extracted 目录的文件。")
    (ART / f"{op}_sample.txt").write_text(demo, encoding="utf-8")
    print(f"完成：best PRM通过率 {best:.3f} → artifacts/{op}_ckpt.pt 等")


if __name__ == "__main__":
    main()
