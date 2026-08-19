# -*- coding: utf-8 -*-
"""Agent 25（后训练研究员）：antNest LLM SFT（监督微调）。

数据：职位问答（32 条）+ 产品知识问答（来自 33/24/28 号交付物）
     + 动作格式示例（任务→```action JSON，教模型输出 Harness 动作协议；
       M6-4 扩至 grep/find 六类工具）
     + M6-1 多轮上下文样本：(任务, 历史, 动作) 三元组，
       与 eval.run_multi_step 观察回填格式一致，攻多步全通率
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
    #    M5-1 扩容：4 条 → 40+ 条（每工具 × 多措辞 × 多对象），攻 L3 工具选择
    #    M6-4 再扩：grep / find 参数化检索工具（动作空间 4 → 6）
    ex += build_action_examples()
    # 4) M6-1 多轮上下文：(任务, 历史, 动作) 三元组，攻多步全通率
    ex += build_multiturn_examples()
    return ex


# ── M5-1：动作示例扩容（工具选择攻坚）──────────────────────
# 每工具多条自然措辞 × 不同对象，教模型「语义 → 工具」而非「字面 → 工具」。
TOOL_LEADS = {
    "list_dir": ["执行列目录：", "我来查看目录内容：", "调用列目录工具："],
    "read_file": ["读取文件：", "我来查看文件内容：", "调用读文件工具："],
    "write_file": ["写入文件：", "我来保存内容：", "调用写文件工具："],
    "shell": ["执行命令统计：", "用 shell 处理：", "调用命令行工具："],
    "grep": ["执行内容检索：", "我来在文件里搜索：", "调用检索工具："],
    "find": ["执行文件查找：", "我来按名称找文件：", "调用查找工具："],
}


def _act(kind, name, args, lead):
    body = (json.dumps({"action": kind, "name": name, "args": args},
                       ensure_ascii=False) if kind == "tool" else
            json.dumps({"action": "finish", "result": "已完成"}, ensure_ascii=False))
    return f"{lead}\n```action\n{body}\n```"


def build_action_examples() -> list:
    """程序化生成动作 SFT 样本：5 类动作 × 8-9 措辞 ≈ 42 条。"""
    specs = [
        # (工具, [(措辞, 对象参数), ...])
        ("list_dir", [
            ("请列出 extracted 目录的文件。", {"p": "/workspace/extracted"}),
            ("看看 extracted 文件夹里都有什么。", {"p": "/workspace/extracted"}),
            ("我想知道 outputs 目录下有哪些交付物。", {"p": "/workspace/antnest_team/outputs"}),
            ("帮我查看 antnest 项目的目录结构。", {"p": "/workspace/antnest"}),
            ("workspace 根目录下有些什么？", {"p": "/workspace"}),
            ("列一下 artifacts 里都存了什么。", {"p": "/workspace/antnest/artifacts"}),
            ("显示 tests 目录的内容。", {"p": "/workspace/antnest/tests"}),
            ("查一下 evals 文件夹里的文件清单。", {"p": "/workspace/antnest/evals"}),
        ]),
        ("read_file", [
            ("读取报告内容。", {"p": "/workspace/antnest/artifacts/report.md"}),
            ("看看报告里写了什么。", {"p": "/workspace/antnest/artifacts/report.md"}),
            ("我想看 README 的内容。", {"p": "/workspace/README.md"}),
            ("打开训练指标文件看看。", {"p": "/workspace/antnest/artifacts/sft_metrics.json"}),
            ("把词表文件内容展示一下。", {"p": "/workspace/antnest/artifacts/v4_vocab.json"}),
            ("查看模型配置。", {"p": "/workspace/antnest/artifacts/v4_model_config.json"}),
            ("读一下样本生成文件。", {"p": "/workspace/antnest/artifacts/v4_sample.txt"}),
            ("这个 md 文件里是什么？看看 memory.md。", {"p": "/workspace/antnest/artifacts/memory.md"}),
        ]),
        ("write_file", [
            ("把结论写成报告。", {"p": "/workspace/antnest/artifacts/report.md",
                                "c": "antNest 报告"}),
            ("把这些材料整理成报告存档。", {"p": "/workspace/antnest/artifacts/report.md",
                                          "c": "antNest 报告"}),
            ("把结论记入 memory.md。", {"p": "/workspace/antnest/artifacts/memory.md",
                                       "c": "结论已记录"}),
            ("帮我保存这些笔记。", {"p": "/workspace/antnest/artifacts/notes.md",
                                  "c": "antNest 笔记"}),
            ("将评测结果写入文件。", {"p": "/workspace/antnest/artifacts/eval_out.md",
                                     "c": "评测结果"}),
            ("生成一份总结文档。", {"p": "/workspace/antnest/artifacts/summary.md",
                                  "c": "antNest 总结"}),
            ("把这段话存成文本文件。", {"p": "/workspace/antnest/artifacts/t.md",
                                      "c": "antNest 文本"}),
            ("输出结果保存到 out.md。", {"p": "/workspace/antnest/artifacts/out.md",
                                        "c": "输出结果"}),
        ]),
        ("shell", [
            ("统计团队交付物数量。", {"cmd": "ls /workspace/antnest_team/outputs | wc -l"}),
            ("数一数 outputs 有多少个文件。", {"cmd": "ls /workspace/antnest_team/outputs | wc -l"}),
            ("帮我查一下词表有多少行。", {"cmd": "wc -l /workspace/antnest/artifacts/v4_vocab.json"}),
            ("搜索文件里的关键词 antNest。", {"cmd": "grep -c antNest /workspace/README.md"}),
            ("统计 README 的字数。", {"cmd": "wc -c /workspace/README.md"}),
            ("列出最近修改的 python 文件。", {"cmd": "find /workspace/antnest -name *.py"}),
            ("看看测试文件有多少个。", {"cmd": "ls /workspace/antnest/tests | wc -l"}),
            ("用命令行查看目录占用。", {"cmd": "ls /workspace/antnest/artifacts"}),
            ("echo 一句口号到终端。", {"cmd": "echo antNest 蚁巢计划"}),
        ]),
        # M6-4：参数化检索工具（动作空间 4 → 6）
        ("grep", [
            ("帮我找找 README 里包含 Harness 的行。", {"p": "/workspace/README.md", "q": "Harness"}),
            ("在词表里搜一下这个词条。", {"p": "/workspace/antnest/artifacts/v4_vocab.json", "q": "蚁巢"}),
            ("报告里哪里提到了沙箱？帮我找出来。", {"p": "/workspace/antnest/artifacts/report.md", "q": "沙箱"}),
            ("在源码文件里查找一下 sandbox。", {"p": "/workspace/antnest/antnest_harness/tools.py", "q": "sandbox"}),
            ("搜一搜笔记里有没有结论两个字。", {"p": "/workspace/antnest/artifacts/notes.md", "q": "结论"}),
            ("看看 README 里哪些行写了蚁巢。", {"p": "/workspace/README.md", "q": "蚁巢"}),
            ("在训练指标里找一下 val_loss。", {"p": "/workspace/antnest/artifacts/v4_metrics.json", "q": "val_loss"}),
            ("帮我检索样本文件里的关键词。", {"p": "/workspace/antnest/artifacts/v4_sample.txt", "q": "antNest"}),
        ]),
        ("find", [
            ("找出仓库里所有 python 文件。", {"dir": "/workspace/antnest", "name": "*.py"}),
            ("查一下 antnest 下的 md 文件有哪些。", {"dir": "/workspace/antnest", "name": "*.md"}),
            ("帮我找出所有 json 配置文件。", {"dir": "/workspace/antnest", "name": "*.json"}),
            ("找一下 tests 目录里的测试文件。", {"dir": "/workspace/antnest/tests", "name": "*.py"}),
            ("仓库里都有哪些词表文件？", {"dir": "/workspace/antnest/artifacts", "name": "*vocab.json"}),
            ("看看有没有 txt 样例文件。", {"dir": "/workspace/antnest/artifacts", "name": "*.txt"}),
            ("递归找一下 evals 下的文件。", {"dir": "/workspace/antnest/evals", "name": "*.json"}),
            ("帮我定位 markdown 交付物。", {"dir": "/workspace/antnest_team/outputs", "name": "*.md"}),
        ]),
    ]
    ex = []
    for tool, utts in specs:
        for i, (q, args) in enumerate(utts):
            lead = TOOL_LEADS[tool][i % len(TOOL_LEADS[tool])]
            ex.append((q, _act("tool", tool, args, lead)))
    # finish 类：多种结束表达
    for q in ["任务已完成。", "任务已完成，请结束。", "所有工作已结束。", "做完了，收工吧。",
              "以上任务全部完成。", "没有更多要做的了。", "到此结束。", "可以结束了。"]:
        ex.append((q, _act("finish", None, None, "任务结束：")))
    return ex


# ── M6-1：多轮上下文样本（(任务, 历史, 动作) 三元组）────────
def multiturn_ctx(task: str, step: int, tool: str) -> str:
    """历史回填格式（与 eval.run_multi_step 逐字一致）。M7-1 起 GRPO 复用。"""
    return (f"{task}\n（第{step}步已完成，使用了 {tool}，结果正常。）"
            f"继续下一步。")


# (任务, [(步号, 该步使用的工具, 期望的下一动作 (kind, tool, args))])
# M7-1：提升为模块级常量，供 GRPO 多步穿透复用（同一训练分布）
MULTITURN_SPECS = [
        ("查看一下 extracted 目录里有什么，随后把文件名清单存档。",
         [(1, "list_dir", ("tool", "write_file",
                           {"p": "/workspace/antnest/artifacts/list.md", "c": "文件清单"})),
          (2, "write_file", ("finish", None, None))]),
        ("先读 README，读完统计它有多少字符。",
         [(1, "read_file", ("tool", "shell",
                            {"cmd": "wc -c /workspace/README.md"})),
          (2, "shell", ("finish", None, None))]),
        ("看看 artifacts 目录的文件，统计文件个数，并把结果记到笔记里。",
         [(1, "list_dir", ("tool", "shell",
                           {"cmd": "ls /workspace/antnest/artifacts | wc -l"})),
          (2, "shell", ("tool", "write_file",
                        {"p": "/workspace/antnest/artifacts/notes.md", "c": "统计结果"})),
          (3, "write_file", ("finish", None, None))]),
        ("瞄一眼 evals 目录，然后数数 json 文件有几个。",
         [(1, "list_dir", ("tool", "shell",
                           {"cmd": "ls /workspace/antnest/evals | wc -l"})),
          (2, "shell", ("finish", None, None))]),
        ("查看词表文件的内容，然后统计它的行数。",
         [(1, "read_file", ("tool", "shell",
                            {"cmd": "wc -l /workspace/antnest/artifacts/v4_vocab.json"})),
          (2, "shell", ("finish", None, None))]),
        ("先看看 extracted 里有什么，再从 README 中检索蚁巢相关的行。",
         [(1, "list_dir", ("tool", "grep",
                           {"p": "/workspace/README.md", "q": "蚁巢"})),
          (2, "grep", ("finish", None, None))]),
        ("读一下 README，再找出仓库里所有 py 文件。",
         [(1, "read_file", ("tool", "find",
                            {"dir": "/workspace/antnest", "name": "*.py"})),
          (2, "find", ("finish", None, None))]),
        ("列出 tests 目录内容，把测试文件清单写进文档。",
         [(1, "list_dir", ("tool", "write_file",
                           {"p": "/workspace/antnest/artifacts/tests_list.md", "c": "测试清单"})),
          (2, "write_file", ("finish", None, None))]),
        ("把训练结论存成报告，然后统计报告字数。",
         [(1, "write_file", ("tool", "shell",
                             {"cmd": "wc -c /workspace/antnest/artifacts/report.md"})),
          (2, "shell", ("finish", None, None))]),
        ("先检索源码里的沙箱实现，再把结果记录下来。",
         [(1, "grep", ("tool", "write_file",
                       {"p": "/workspace/antnest/artifacts/grep_out.md", "c": "检索结果"})),
          (2, "write_file", ("finish", None, None))]),
]


def build_multiturn_examples() -> list:
    """多步任务的多轮样本：上下文 = 任务 + 已完成步骤的历史摘要，
    与 eval.run_multi_step 的观察回填格式逐字一致（第N步已完成，使用了
    X，结果正常。继续下一步。），教模型依据历史选出正确的下一步动作。

    训练任务措辞与评测集 multi_tasks 物理隔离（语义同类、字面不同），
    多步全通率为 0 的根因即训练分布中从未出现过带历史的上下文。
    """
    ex = []
    for task, trans in MULTITURN_SPECS:
        for j, (step, used, (kind, tool, args)) in enumerate(trans):
            q = multiturn_ctx(task, step, used)
            if kind == "finish":
                ex.append((q, _act("finish", None, None, "任务结束：")))
            else:
                lead = TOOL_LEADS[tool][j % len(TOOL_LEADS[tool])]
                ex.append((q, _act("tool", tool, args, lead)))
    return ex


def load_traj_examples(path) -> list:
    """M4 轨迹回流：Harness 真实工具调用轨迹 → SFT 样本（"使用即训练"）。

    成功轨迹（ok=True 且含成功调用）→ (任务, 首个成功动作的围栏表达)，
    把线上真实用法蒸馏回训练分布。
    """
    p = Path(path)
    if not p.exists():
        return []
    ex = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not t.get("ok"):
            continue
        good = [c for c in t.get("calls", []) if c.get("ok")]
        if not good:
            continue
        c = good[0]
        a = ('执行：\n```action\n' + json.dumps(
            {"action": "tool", "name": c["name"], "args": c.get("args", {})},
            ensure_ascii=False) + '\n```')
        ex.append((t["task"], a))
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


def load_sft_dataset(tok, block, seed=42, traj_path=""):
    ex = build_examples()
    if traj_path:
        ex += load_traj_examples(traj_path)
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
    ap.add_argument("--traj", default="", help="M4 轨迹回流：trajs.jsonl 路径")
    args = ap.parse_args()

    torch.manual_seed(7)
    bp, op = args.base_prefix, args.out_prefix
    cfg = json.loads((ART / f"{bp}_model_config.json").read_text(encoding="utf-8"))
    block = min(args.block, cfg["block_size"])  # QA 修复①：SFT 序列不得超过模型 ctx
    # QA 修复②：词表版本锁定——必须用基座训练时保存的词表，避免语料演进导致词表漂移
    tok = load_tokenizer(ART / f"{bp}_vocab.json")
    assert len(tok) == cfg["vocab"], \
        f"词表不匹配: {len(tok)} != {cfg['vocab']}（checkpoint 与词表版本必须一致）"
    train, val = load_sft_dataset(tok, block, traj_path=args.traj)
    print(f"SFT 数据：{len(train)} 训练 / {len(val)} 验证（共 {len(train)+len(val)} 条指令，"
          f"ctx={block}）" + (f"，含轨迹回流" if args.traj else ""))

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
