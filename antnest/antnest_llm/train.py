# -*- coding: utf-8 -*-
"""Agent 12 + Agent 24：antNest LLM 预训练入口（M1 版）。

M1 升级（落实 23 号《算法-系统联合优化意见》）：
  - 语料 v2（corpus.py）：JD + 团队交付物 + 产品源码
  - warmup + 余弦退火学习率（替代恒定 lr）
  - 验证集择优保存（save best-val checkpoint）
  - 模型配置持久化（model_config_*.json，供 Harness 直连加载）

用法：
  python -m antnest_llm.train --corpus v2 --steps 240 --prefix v2
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from .tokenizer import CharTokenizer, load_corpus
from .corpus import load_corpus_v2, load_corpus_v3, load_corpus_v4, load_corpus_v5
from .bpe import BPETokenizer
from .model import TinyGPT
from .data import build_dataset, get_batch

ART = Path(__file__).resolve().parent.parent / "artifacts"


def lr_at(step, total, base_lr, warmup=20):
    """warmup 线性升温 + 余弦退火至 10% 基准。"""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n_embd", type=int, default=96)
    ap.add_argument("--n_head", type=int, default=3)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--corpus", choices=["v1", "v2", "v3", "v4", "v5"], default="v1")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--bpe_vocab", type=int, default=2200)
    ap.add_argument("--prefix", default="", help="产物文件名前缀，如 v2")
    ap.add_argument("--resume", default="", help="从该 checkpoint 续训")
    args = ap.parse_args()

    torch.manual_seed(1337)
    ART.mkdir(exist_ok=True)
    pfx = f"{args.prefix}_" if args.prefix else ""

    # ── 数据（Agent 3）──
    loader = {"v1": load_corpus, "v2": load_corpus_v2, "v3": load_corpus_v3,
              "v4": load_corpus_v4, "v5": load_corpus_v5}[args.corpus]
    corpus = loader()
    if args.tokenizer == "bpe":
        tok = BPETokenizer.train(corpus, args.bpe_vocab)
    else:
        tok = CharTokenizer(corpus)
    tok.save(ART / f"{pfx}vocab.json")
    train_ids, val_ids = build_dataset(tok, args.block)
    print(f"语料[{args.corpus}] {len(corpus)} 字符 | 词表 {len(tok)} | "
          f"训练 {len(train_ids)} / 验证 {len(val_ids)} tokens")

    # ── 模型（Agent 24）──
    model = TinyGPT(len(tok), args.n_embd, args.n_head, args.n_layer, args.block)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, weights_only=True))
        print(f"续训自 {args.resume}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyGPT 参数量: {n_params/1e3:.1f}K "
          f"({args.n_layer}L-{args.n_embd}d-{args.n_head}h, ctx={args.block})")

    # ── 训练（Agent 12）：warmup+余弦 & 择优保存 ──
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history, t0, best_val = [], time.time(), float("inf")
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)
        x, y = get_batch(train_ids, val_ids, args.block, args.batch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0 or step == 1:
            model.eval()
            vx, vy = get_batch(train_ids, val_ids, args.block, args.batch, train=False)
            with torch.no_grad():
                _, vloss = model(vx, vy)
            model.train()
            vl = vloss.item()
            history.append({"step": step, "train_loss": round(loss.item(), 4),
                            "val_loss": round(vl, 4), "lr": round(lr_at(step, args.steps, args.lr), 5)})
            star = ""
            if vl < best_val:
                best_val = vl
                torch.save(model.state_dict(), ART / f"{pfx}ckpt.pt")
                star = " *best"
            print(f"step {step:>4} | train {loss.item():.4f} | val {vl:.4f}{star}")

    # ── 产出（Agent 14 可观测）──
    (ART / f"{pfx}metrics.json").write_text(json.dumps(
        {"config": vars(args), "params": n_params, "corpus_chars": len(corpus),
         "vocab": len(tok), "best_val_loss": round(best_val, 4),
         "seconds": round(time.time() - t0, 1), "history": history},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (ART / f"{pfx}model_config.json").write_text(json.dumps(
        {"n_embd": args.n_embd, "n_head": args.n_head, "n_layer": args.n_layer,
         "block_size": args.block, "vocab": len(tok)}, ensure_ascii=False), encoding="utf-8")

    model.eval()
    prompt_ids = tok.encode("模型") or [0]
    out = model.generate(torch.tensor([prompt_ids], dtype=torch.long),
                         max_new_tokens=150)[0].tolist()
    sample = tok.decode(out)
    (ART / f"{pfx}sample.txt").write_text(sample, encoding="utf-8")
    print(f"完成：{args.steps} 步 / {time.time()-t0:.0f}s | best val {best_val:.4f} "
          f"→ artifacts/{pfx}ckpt.pt 等")
    print("采样预览：", sample[:80].replace("\n", " "))


if __name__ == "__main__":
    main()
