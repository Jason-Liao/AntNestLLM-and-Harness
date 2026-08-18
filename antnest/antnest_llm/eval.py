# -*- coding: utf-8 -*-
"""Agent 19（评测数据工程师）：评测与训练数据分离（M4 扩容版）。

原则：评测集永远不进训练梯度。
  - evals/evalset.json 独立维护（24 动作 + 12 QA），与训练任务池物理隔离
  - 同一把严格尺子评任意 checkpoint（v3/sft3/grpo/v4/sft4/grpo5…）
  - M4：评分升级到工具级（L3 选对工具才给满分）+ pass@k 采样评测
    （pass@1=贪心；pass@k=温度采样 k 次，任一次达到阈值即通过）

用法：python -m antnest_llm.eval --ckpt grpo5 --passk 3
"""
import argparse
import json
import re
from pathlib import Path

import torch

from .bpe import load_tokenizer
from .model import TinyGPT
from .sft import U, A, greedy_answer

ART = Path(__file__).resolve().parent.parent / "artifacts"
EVALSET = Path(__file__).resolve().parent.parent / "evals" / "evalset.json"

LEGAL_TOOLS = {"list_dir", "shell", "write_file", "read_file"}
# 通过阈值：L1+L2+L3=0.7（格式对 + 类型对 + 选对工具）
PASS_BAR = 0.7


def score_action(text: str, expect: str, tool: str = None) -> float:
    """严格尺子：L1 格式 0.3 / L2 类型 0.2 / L3 工具选择 0.3 / L4 参数 0.2。"""
    m = re.search(r"```action\s*(\{.*?\})\s*```", text, re.S)
    if not m:
        return 0.0
    s = 0.3
    try:
        act = json.loads(m.group(1))
    except json.JSONDecodeError:
        return s
    if act.get("action") not in ("tool", "finish"):
        return s
    if act.get("action") != expect:
        return s
    s += 0.2
    if expect == "finish":
        return s + (0.5 if str(act.get("result", "")).strip() else 0.0)
    name = act.get("name")
    if tool is not None:
        if name != tool:
            return s
        s += 0.3
    elif name in LEGAL_TOOLS:
        s += 0.3
    args = act.get("args")
    if isinstance(args, dict) and (name == "finish" or args):
        s += 0.2
    return s


def score_qa(answer: str, keys: list) -> float:
    """关键词命中率（参考答案关键词在生成文本中的覆盖比例）。"""
    hit = sum(1 for k in keys if k in answer)
    return round(hit / len(keys), 4) if keys else 0.0


@torch.no_grad()
def sample_answer(model, tok, q: str, max_new=120, temperature=0.8, top_k=20):
    """温度采样生成（pass@k 用）。"""
    ids = tok.encode(f"{U}{q}\n{A}") or [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=max_new,
                         temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


@torch.no_grad()
def evaluate(prefix: str, passk: int = 1, verbose=True):
    cfg = json.loads((ART / f"{prefix}_model_config.json").read_text(encoding="utf-8"))
    tok = load_tokenizer(ART / f"{prefix}_vocab.json")
    assert len(tok) == cfg["vocab"], f"{prefix}: 词表与 checkpoint 不一致"
    model = TinyGPT(cfg["vocab"], cfg["n_embd"], cfg["n_head"],
                    cfg["n_layer"], cfg["block_size"])
    model.load_state_dict(torch.load(ART / f"{prefix}_ckpt.pt", weights_only=True))
    model.eval()

    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    act_scores, qa_scores, pass_hits = [], [], []
    for t in es["action_tasks"]:
        ans = greedy_answer(model, tok, t["prompt"], max_new=120)
        sc = score_action(ans, t["expect"], t.get("tool"))
        act_scores.append(sc)
        # pass@k：贪心未达标时再采 k-1 次补测（k=1 即纯贪心）
        ok = sc >= PASS_BAR
        if not ok and passk > 1:
            for _ in range(passk - 1):
                s = score_action(sample_answer(model, tok, t["prompt"]),
                                 t["expect"], t.get("tool"))
                if s >= PASS_BAR:
                    ok = True
                    break
        pass_hits.append(1.0 if ok else 0.0)
    for t in es["qa_tasks"]:
        ans = greedy_answer(model, tok, t["prompt"], max_new=120)
        qa_scores.append(score_qa(ans, t["keys"]))

    res = {
        "ckpt": prefix,
        "action_pass": round(sum(act_scores) / len(act_scores), 4),
        f"pass@{passk}": round(sum(pass_hits) / len(pass_hits), 4),
        "qa_hit": round(sum(qa_scores) / len(qa_scores), 4),
        "overall": round((sum(act_scores) + sum(qa_scores))
                         / (len(act_scores) + len(qa_scores)), 4),
        "n_tasks": len(es["action_tasks"]) + len(es["qa_tasks"]),
    }
    if verbose:
        print(f"[{prefix}] 动作 {res['action_pass']:.3f} | pass@{passk} "
              f"{res[f'pass@{passk}']:.3f} | QA {res['qa_hit']:.3f} | "
              f"综合 {res['overall']:.3f}（{res['n_tasks']} 任务）")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="grpo5", help="checkpoint 前缀（可逗号分隔多个）")
    ap.add_argument("--passk", type=int, default=1, help="pass@k 的 k（1=纯贪心）")
    args = ap.parse_args()
    out = []
    for p in args.ckpt.split(","):
        p = p.strip()
        if p:
            out.append(evaluate(p, passk=args.passk))
    if len(out) > 1:
        (ART / "eval_compare.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("对比结果 → artifacts/eval_compare.json")


if __name__ == "__main__":
    main()
