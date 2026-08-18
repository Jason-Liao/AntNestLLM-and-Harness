# -*- coding: utf-8 -*-
"""antNest LLM —— 蚁巢大模型训练与推理包（PyTorch CPU 参考实现）。

分工：
  Agent 3  预训练数据工程师 : 数据采集与字符 tokenizer
  Agent 24 预训练研究员     : 模型结构（Decoder-only GPT）
  Agent 12 训推框架工程师   : 训练循环 / 采样推理
"""
from .tokenizer import CharTokenizer
from .model import TinyGPT, TransformerBlock
from . import data
