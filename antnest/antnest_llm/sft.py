# -*- coding: utf-8 -*-
"""Agent 25（后训练研究员）：antNest LLM SFT（监督微调）。

数据：职位问答（32 条）+ 产品知识问答（来自 33/24/28 号交付物）
     + 动作格式示例（任务→```action JSON，教模型输出 Harness 动作协议）
训练：assistant 区间损失（prompt 部分 -100 掩码），warmup+余弦，择优保存。

用法：python -m antnest_llm.sft --steps 200
产出：artifacts/{sft_ckpt.pt, sft_metrics.json, sft_model_config.json,
                sft_vocab.json, sft_sample.txt}
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .bpe import load_tokenizer
from .corpus import jd_titles
from .model import TinyGPT
from .train import lr_at

ART = Path(__file__).resolve().parent.parent / "artifacts"
U, A = "<|user|>", "<|assistant|>"


# ── 数据构造（Agent 25 + Agent 18/19 的评测素材转化）──────────
def build_examples() -> list:
    ex = []

    # 1) 职位职责问答（Agent 3 语料 → QA）
    for num, title in jd_titles():
        q = f"antNest 团队中「{title}」的使命是什么？"
        a = (f"「{title}」是 antNest 蚁巢计划的第 {num} 号职位。"
             f"其使命围绕 antNest LLM 训练与 antNest Harness 建设展开，"
             f"以工程与创新共同推进 AGI 目标。")
        ex.append((q, a))

    # 2) 产品知识问答（取自真实交付物数据）
    knowledge = [
        ("antNest 的目标是什么？",
         "antNest（蚁巢）模型计划有两个目标：训练出 antNest LLM，并打造 antNest Harness。"
         "模型与外壳相互驱动、共同进化。"),
        ("antNest Harness 包含哪些机制？",
         "antNest Harness 包含 Agent Loop、Tool Use（DSec 沙箱）、记忆与上下文管理、"
         "Subagent 与 Multi-Agent 协作，LLM 可插拔直连。"),
        ("antNest LLM 用什么结构？",
         "antNest LLM 采用 Decoder-only Transformer（TinyGPT），因果自注意力，"
         "在自有语料上预训练后经 SFT 对齐动作协议。"),
        ("质量保障怎么做？",
         "antNest 用 pytest 建立质量门禁，覆盖 tokenizer、模型收敛、沙箱安全、"
         "Agent Loop 与 Multi-Agent，并执行 badcase 定位-修复-回归闭环。"),
    ]
    ex += knowledge

    # 3) 动作格式（Agent 5 的动作协议 → 训练信号）
    actions = [
        ("请列出 extracted 目录的文件。",
         '执行列目录：\n```action\n{"action":"tool","name":"list_dir","args":{"p":"/workspace/extracted"}}\n```'),
        ("统计团队交付物数量。",
         '用 shell 统计：\n```action\n{"action":"tool","name":"shell","args":{"cmd":"ls /workspace/antnest_team/outputs | wc -l"}}\n```'),
        ("把结论写成报告。",
         '写入报告文件：\n```action\n{"action":"tool","name":"write_file","args":{"p":"/workspace/antnest/artifacts/report.md","c":"antNest 报告"}}\n```'),
        ("任务已完成。", '任务结束：\n```action\n{"action":"finish","result":"已完成"}\n```'),
    ]
    ex += actions
    return ex


def encode_example(tok, q, a, block):
    """返回 (x, y)，y 在 prompt 区间为 -100（仅监督 assistant 部分）。"""
    prompt = f"{U}{q}\n{A}"
    full = prompt + a
    ids = tok.encode(full)
    if len(ids) < 2:
        return None
    ids = ids[: block + 1]
    p_len = min(len(tok.encode(prompt)), len(ids) - 1)
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    y[: p_len - 1] = -100  # y[i] 对应 token i+1，prompt 区间全部掩码
    return x, y


def load_sft_dataset(tok, block, seed=42):
    ex = build_examples()
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ex), generator=rng).tolist()
    n_val = max(2, len(ex) // 10)
    val, train = idx[:n_val], idx[n_val:]
    enc = lambda ids: [e for e in (encode_example(tok, *ex[i], block) for i in ids) if e]
    return enc(train), enc(val)


def run_epoch(model, data, opt=None, batch=16):
    """一个 epoch；opt 为 None 时仅评估（返回平均 masked loss）。"""
    tot, cnt = 0.0, 0
    for i in range(0, len(data), batch):  # QA 修复③：允许末尾小批，验证集<batch 时不再空转返回 0
        xs = torch.nn.utils.rnn.pad_sequence(
            [x for x, _ in data[i:i + batch]], batch_first=True)
        ys = torch.nn.utils.rnn.pad_sequence(
            [y for _, y in data[i:i + batch]], batch_first=True,
            padding_value=-100)
        T = xs.size(1)
        pos = torch.arange(T)
        x, y = xs, ys
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=-100)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += loss.item(); cnt += 1
    return tot / max(1, cnt)


@torch.no_grad()
def greedy_answer(model, tok, q, max_new=100):
    model.eval()
    ids = tok.encode(f"{U}{q}\n{A}") or [0]
    idx = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.block_size:])
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        if tok.decode(nxt[0].tolist()).endswith("\n\n"):
            break
    return tok.decode(idx[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200, help="epoch 数")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--base_prefix", default="v2", help="基座产物前缀（v2/v3）")
    ap.add_argument("--out_prefix", default="sft", help="SFT 产物前缀")
    ap.add_argument("--base_ckpt", default="", help="覆盖默认基座 checkpoint 路径")
    args = ap.parse_args()

    torch.manual_seed(7)
    bp, op = args.base_prefix, args.out_prefix
    cfg = json.loads((ART / f"{bp}_model_config.json").read_text(encoding="utf-8"))
    block = min(args.block, cfg["block_size"])  # QA 修复①：SFT 序列不得超过模型 ctx
    # QA 修复②：词表版本锁定——必须用基座训练时保存的词表，避免语料演进导致词表漂移
    tok = load_tokenizer(ART / f"{bp}_vocab.json")
    assert len(tok) == cfg["vocab"], \
        f"词表不匹配: {len(tok)} != {cfg['vocab']}（checkpoint 与词表版本必须一致）"
    train, val = load_sft_dataset(tok, block)
    print(f"SFT 数据：{len(train)} 训练 / {len(val)} 验证（共 {len(train)+len(val)} 条指令，"
          f"ctx={block}）")

    model = TinyGPT(cfg["vocab"], cfg["n_embd"], cfg["n_head"],
                    cfg["n_layer"], cfg["block_size"])
    base_ckpt = args.base_ckpt or str(ART / f"{bp}_ckpt.pt")
    model.load_state_dict(torch.load(base_ckpt, weights_only=True))
    print(f"自 {base_ckpt} 续训（{sum(p.numel() for p in model.parameters())/1e3:.1f}K 参数）")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history, t0, best = [], time.time(), float("inf")
    for ep in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep, args.steps, args.lr)
        tr = run_epoch(model, train, opt)
        if ep % 10 == 0 or ep == 1:
            vl = run_epoch(model, val)
            history.append({"epoch": ep, "train_loss": round(tr, 4),
                            "val_loss": round(vl, 4)})
            star = ""
            if vl < best:
                best = vl
                torch.save(model.state_dict(), ART / f"{op}_ckpt.pt")
                star = " *best"
            print(f"epoch {ep:>3} | train {tr:.4f} | val {vl:.4f}{star}")

    tok.save(ART / f"{op}_vocab.json")
    (ART / f"{op}_model_config.json").write_text(
        (ART / f"{bp}_model_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    (ART / f"{op}_metrics.json").write_text(json.dumps(
        {"steps": args.steps, "n_train": len(train), "n_val": len(val),
         "best_val_loss": round(best, 4), "seconds": round(time.time() - t0, 1),
         "history": history}, ensure_ascii=False, indent=1), encoding="utf-8")
    demo_q = "antNest 的目标是什么？"
    ans = greedy_answer(model, tok, demo_q)
    (ART / f"{op}_sample.txt").write_text(ans, encoding="utf-8")
    print(f"完成：best val {best:.4f} → artifacts/{op}_ckpt.pt 等")
    print("问答采样：", ans.replace("\n", " ")[:100])


if __name__ == "__main__":
    main()
