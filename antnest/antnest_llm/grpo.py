# -*- coding: utf-8 -*-
"""Agent 18/19 + Agent 25：antNest LLM RL 环境（GRPO）。

GRPO（Group Relative Policy Optimization）——以"评测通过率"为奖励：
  1) 对每个 prompt 采样一组（group）响应
  2) 奖励函数按 Harness 动作协议评测：格式可解析 / 动作合法 / 参数完备
  3) 组内归一化优势 A=(r-mean)/(std+eps)，替代 PPO 的 critic
  4) 裁剪 policy gradient + 冻结参考模型的 KL 惩罚，防止崩塌

用法：python -m antnest_llm.grpo --iters 30
产出：artifacts/{grpo_ckpt.pt, grpo_metrics.json, grpo_model_config.json,
                grpo_vocab.json}
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
from .sft import U, A, greedy_answer

ART = Path(__file__).resolve().parent.parent / "artifacts"

# ── 评测环境（Agent 18：任务定义；Agent 19：奖励即评测）──────────────
# 合法动作协议（与 antnest_harness/agent.py 的 ACTION 协议一致）
LEGAL_TOOLS = {"list_dir", "shell", "write_file", "read_file"}

# 任务池：prompt → 期望动作类型（评测参考答案，不进入训练梯度）
TASK_POOL = [
    ("请列出 extracted 目录的文件。", "tool"),
    ("查看 /workspace 下的交付物目录。", "tool"),
    ("统计团队交付物数量。", "tool"),
    ("把结论写成报告。", "tool"),
    ("读取报告内容。", "tool"),
    ("任务已完成，请结束。", "finish"),
    ("所有工作已结束。", "finish"),
    ("目标达成了。", "finish"),
]


def evaluate_response(text: str, expect: str) -> float:
    """训练奖励 = 严格通过率 + 稀疏 shaping（引导 mini 模型从零起步）。

    注：训练奖励允许塑形（reward shaping），对外评测仍以 eval.py 的
    严格尺子为准——这正是"评测与训练数据分离"的另一层含义。
    """
    score = 0.0
    # ① 动作围栏（0.3）：```action ... ``` 存在
    m = re.search(r"```action\s*(\{.*?\})\s*```", text, re.S)
    if not m:
        # shaping：格式要素部分分，让奖励信号不再全 0
        if "```action" in text:
            score += 0.06
        if '"action"' in text:
            score += 0.06
        if '"tool"' in text or '"finish"' in text:
            score += 0.06
        return score
    score += 0.3
    # ② JSON 可解析 + 动作合法（0.4）
    try:
        act = json.loads(m.group(1))
    except json.JSONDecodeError:
        return score
    kind = act.get("action")
    if kind not in ("tool", "finish"):
        return score
    score += 0.2
    if kind != expect:
        return score  # 类型对了才有后续分
    score += 0.2
    # ③ 参数完备（0.3）：tool→name∈LEGAL_TOOLS 且 args 是 dict；finish→result 非空
    if kind == "tool":
        if act.get("name") in LEGAL_TOOLS and isinstance(act.get("args"), dict):
            score += 0.3
    else:
        if str(act.get("result", "")).strip():
            score += 0.3
    return score


@torch.no_grad()
def sample_group(model, tok, prompt: str, g: int, max_new: int, temperature: float):
    """对同一 prompt 采样 g 个响应，返回 (ids 列表, logprob 前缀长度)。"""
    pids = tok.encode(f"{U}{prompt}\n{A}") or [0]
    ctx = model.block_size
    idx = torch.tensor([pids[-ctx:]] * g, dtype=torch.long)
    p_len = idx.size(1)
    for _ in range(max_new):
        logits, _ = model(idx[:, -ctx:])
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        # 终止判定：所有序列都出现 "\n\n" 或已到 ctx
        idx = torch.cat([idx, nxt], dim=1)
        if idx.size(1) >= ctx:
            break
        new_txt = tok.decode(idx[0, p_len:].tolist())
        if all(tok.decode(r.tolist()).endswith("```") or "\n\n" in tok.decode(r.tolist())
               for r in idx[:, p_len + 8:]):
            break
    return [r.tolist() for r in idx], p_len


def seq_logprob(model, ids: list, p_len: int):
    """响应区间的 token logprob 总和（含梯度）。"""
    s = max(p_len, 1)  # 防止 p_len=0 时 -1 索引到序列末尾
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor(ids[s:], dtype=torch.long)  # 一维 target
    if y.numel() == 0 or x.size(1) == 0:
        return None, None
    logits, _ = model(x)
    n = min(y.numel(), logits.size(1) - (s - 1))
    if n <= 0:
        return None, None
    # cross_entropy(reduction=none) 即逐 token NLL，负号得 logprob
    nll = F.cross_entropy(logits[0, s - 1: s - 1 + n], y[:n], reduction="none")
    return -nll.sum(), n


def grpo_loss(model, ref, ids: list, p_len: int, adv: float, clip=0.2, beta=0.05):
    """单条响应的 GRPO 目标：-min(rA, clip(r)A) + beta·KL(r‖ref)。"""
    lp, n = seq_logprob(model, ids, p_len)
    if lp is None:
        return None
    with torch.no_grad():
        rlp, rn = seq_logprob(ref, ids, p_len)
        if rlp is None:
            return None
    # 长度归一化 logratio（per-token IS），再钳制防溢出（QA 修复：loss 3.5e8 爆炸）
    logratio = (lp / n) - (rlp / rn)
    ratio = torch.exp(torch.clamp(logratio, -5.0, 5.0))
    un = -adv * ratio
    cl = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    loss = torch.max(un, cl)
    # 逐 token KL 近似（k3 估计）
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--group", type=int, default=6, help="组大小 G")
    ap.add_argument("--prompts_per_iter", type=int, default=4)
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
    rng = torch.Generator().manual_seed(3)
    history, t0, best = [], time.time(), -1.0
    tasks = TASK_POOL

    for it in range(1, args.iters + 1):
        sel = [tasks[i] for i in torch.randperm(len(tasks), generator=rng)
               [: args.prompts_per_iter]]
        tot_loss, tot_rew, n_g = 0.0, 0.0, 0
        for prompt, expect in sel:
            model.eval()
            group, p_len = sample_group(model, tok, prompt, args.group,
                                        args.max_new, args.temperature)
            rewards = [evaluate_response(tok.decode(g[p_len:]), expect) for g in group]
            r_t = torch.tensor(rewards)
            adv = ((r_t - r_t.mean()) / (r_t.std(unbiased=False) + 1e-4)).tolist()
            tot_rew += r_t.mean().item(); n_g += 1
            # QA 修复：组内聚合一次 step（逐条 step 更新过频导致 policy 崩塌）
            model.train()
            g_losses = [l for l in (grpo_loss(model, ref, g_ids, p_len, a)
                                    for g_ids, a in zip(group, adv)) if l is not None]
            if not g_losses:
                continue
            loss_t = torch.stack(g_losses).mean()
            opt.zero_grad(set_to_none=True)
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            tot_loss += loss_t.item()
        mean_rew = tot_rew / max(1, n_g)
        history.append({"iter": it, "loss": round(tot_loss, 4),
                        "reward": round(mean_rew, 4)})
        if it % 5 == 0 or it == 1:
            print(f"iter {it:>3} | loss {tot_loss:>8.3f} | 平均通过率 {mean_rew:.3f}")
        if mean_rew >= best:
            best = mean_rew
            torch.save(model.state_dict(), ART / f"{op}_ckpt.pt")

    tok.save(ART / f"{op}_vocab.json")
    (ART / f"{op}_model_config.json").write_text(
        (ART / f"{bp}_model_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    (ART / f"{op}_metrics.json").write_text(json.dumps(
        {"algo": "GRPO", "iters": args.iters, "group": args.group,
         "best_reward": round(best, 4), "final_reward": history[-1]["reward"],
         "seconds": round(time.time() - t0, 1), "history": history},
        ensure_ascii=False, indent=1), encoding="utf-8")
    model.eval()
    demo = greedy_answer(model, tok, "请列出 extracted 目录的文件。")
    (ART / "grpo_sample.txt").write_text(demo, encoding="utf-8")
    print(f"完成：best 通过率 {best:.3f} → artifacts/{op}_ckpt.pt 等")


if __name__ == "__main__":
    main()
