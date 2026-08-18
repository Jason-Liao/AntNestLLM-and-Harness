# -*- coding: utf-8 -*-
"""Agent 19（评测数据工程师）：评测与训练数据分离。

M3 原则：评测集永远不进训练梯度。
  - evalset.json 独立维护，与 sft.py/grpo.py 的训练任务池物理隔离
  - eval.py 支持任意 checkpoint（v3/sft3/grpo）在同一把尺子下对比
  - 指标：动作通过率（格式/类型/参数）+ QA 检索命中率（关键词命中）

用法：python -m antnest_llm.eval --ckpt grpo
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


def score_action(text: str, expect: str) -> float:
    """与 grpo.evaluate_response 同尺（评测代码复用，任务数据分离）。"""
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
    s += 0.2
    if act.get("action") != expect:
        return s
    s += 0.2
    if expect == "tool":
        if act.get("name") in LEGAL_TOOLS and isinstance(act.get("args"), dict):
            s += 0.3
    elif str(act.get("result", "")).strip():
        s += 0.3
    return s


def score_qa(answer: str, keys: list) -> float:
    """关键词命中率（参考答案关键词在生成文本中的覆盖比例）。"""
    hit = sum(1 for k in keys if k in answer)
    return round(hit / len(keys), 4) if keys else 0.0


@torch.no_grad()
def evaluate(prefix: str, verbose=True):
    cfg = json.loads((ART / f"{prefix}_model_config.json").read_text(encoding="utf-8"))
    tok = load_tokenizer(ART / f"{prefix}_vocab.json")
    assert len(tok) == cfg["vocab"], f"{prefix}: 词表与 checkpoint 不一致"
    model = TinyGPT(cfg["vocab"], cfg["n_embd"], cfg["n_head"],
                    cfg["n_layer"], cfg["block_size"])
    model.load_state_dict(torch.load(ART / f"{prefix}_ckpt.pt", weights_only=True))
    model.eval()

    es = json.loads(EVALSET.read_text(encoding="utf-8"))
    act_scores, qa_scores, rows = [], [], []
    for t in es["action_tasks"]:
        ans = greedy_answer(model, tok, t["prompt"], max_new=120)
        sc = score_action(ans, t["expect"])
        act_scores.append(sc)
        rows.append((t["prompt"][:18], sc))
    for t in es["qa_tasks"]:
        ans = greedy_answer(model, tok, t["prompt"], max_new=120)
        sc = score_qa(ans, t["keys"])
        qa_scores.append(sc)
        rows.append((t["prompt"][:18], sc))

    res = {
        "ckpt": prefix,
        "action_pass": round(sum(act_scores) / len(act_scores), 4),
        "qa_hit": round(sum(qa_scores) / len(qa_scores), 4),
        "overall": round((sum(act_scores) + sum(qa_scores))
                         / (len(act_scores) + len(qa_scores)), 4),
    }
    if verbose:
        print(f"[{prefix}] 动作通过率 {res['action_pass']:.3f} | "
              f"QA 命中率 {res['qa_hit']:.3f} | 综合 {res['overall']:.3f}")
        for name, sc in rows:
            print(f"   {sc:.2f}  {name}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="grpo", help="checkpoint 前缀（可逗号分隔多个）")
    args = ap.parse_args()
    out = []
    for p in args.ckpt.split(","):
        p = p.strip()
        if p:
            out.append(evaluate(p))
    if len(out) > 1:
        (ART / "eval_compare.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("对比结果 → artifacts/eval_compare.json")


if __name__ == "__main__":
    main()
