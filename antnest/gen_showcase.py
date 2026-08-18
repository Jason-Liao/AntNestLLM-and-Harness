# -*- coding: utf-8 -*-
"""antNest LLM 生成能力实测：三个 checkpoint 对比采样。"""
import json
from pathlib import Path

import torch

from antnest_llm.tokenizer import CharTokenizer
from antnest_llm.model import TinyGPT

ART = Path("/workspace/antnest/artifacts")


def load(pfx):
    sep = "_" if pfx else ""
    cfg_file = ART / f"{pfx}{sep}model_config.json"
    tok = CharTokenizer.load(str(ART / f"{pfx}{sep}vocab.json"))
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    else:  # M0 旧版产物无 config，按默认超参重建
        cfg = {"n_embd": 96, "n_head": 3, "n_layer": 3,
               "block_size": 128, "vocab": len(tok)}
    m = TinyGPT(cfg["vocab"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["block_size"])
    m.load_state_dict(torch.load(ART / f"{pfx}{sep}ckpt.pt", weights_only=True))
    m.eval()
    return m, tok


def gen(m, tok, prompt, n=120, temp=0.7):
    ids = tok.encode(prompt) or [0]
    out = m.generate(torch.tensor([ids], dtype=torch.long),
                     max_new_tokens=n, temperature=temp, top_k=20)
    return tok.decode(out[0].tolist())


torch.manual_seed(7)
for name, pfx, prompt in [
    ("M0 基座(300步)", "", "团队"),
    ("v2 预训练(240步)", "v2", "训练"),
    ("SFT 指令微调", "sft", "<|user|>antNest 的目标是什么？\n<|assistant|>"),
]:
    m, tok = load(pfx)
    print(f"\n══ {name} | prompt: {prompt!r}")
    print(gen(m, tok, prompt))
