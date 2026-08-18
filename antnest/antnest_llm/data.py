# -*- coding: utf-8 -*-
"""Agent 3（预训练数据工程师）：数据集切分与批采样。"""
import torch
from .tokenizer import load_corpus


def build_dataset(tokenizer, block_size=128):
    """全量语料编码后按 95%/5% 切分训练/验证集。"""
    ids = torch.tensor(tokenizer.encode(load_corpus()), dtype=torch.long)
    n = int(len(ids) * 0.95)
    return ids[:n], ids[n:]


def get_batch(split_ids, val_ids, block_size=128, batch_size=16, train=True):
    data = split_ids if train else val_ids
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
