# AntNestLLM-and-Harness 🐜

> antNest（蚁巢）模型计划：由 **32 个智能体组成的虚拟团队**，端到端训练 **antNest LLM**（自研大模型）并打造 **antNest Harness**（智能体外壳）——让"模型"与"外壳"相互驱动、共同进化。

---

## 一、这个项目是怎么来的

一切从 32 份真实职位说明（JD）开始：

1. **蚁巢团队组建**：工作区根目录的 `1.服务端开发工程师….md` 到 `32.行政团队.md` 共 32 份职位文件（含 `.doc` 二进制，已通过 `extract_text.py` 提取为 `extracted/*.md`），覆盖 AI 产品、预训练/后训练研究员、Agent Harness、Agent Infra、数据工程、评测、测试、HR/法务/财务等 32 个职位。
2. **32 Agent 一一映射**：`antnest_team/antnest_agents.py` 把 32 个职位定义为 32 个 CrewAI Agent（角色/目标/背景故事均取自 JD 原文），`antnest_tasks.py` 编排 33 个任务形成产品开发流水线，`main.py` 是团队入口。
3. **"蚁巢用自己铸造自己"**：团队首先造出最小可用的 LLM 和 Harness，随后每一轮迭代都把**团队自己的交付物、产品源码、工具调用轨迹**回灌为训练语料——模型越用越强，外壳越跑越稳，形成数据自增长闭环。

七个里程碑（每个冲刺报告见 `antnest_team/outputs/`）：

| 里程碑 | 主题 | 关键产出 |
|---|---|---|
| M0/M1 | 从 0 到 1 | TinyGPT 预训练管线、Harness Agent Loop / 沙箱工具 / Multi-Agent、19 项测试 |
| M2 | 数据与分词升级 | 语料清洗器、BPE 分词器、v3 预训练 + SFT、本地模型对话模式 |
| M3 | 强化学习闭环 | GRPO（评测通过率为奖励）、评测与训练数据分离、v4 模型放大到 7.7M、ChatCrew 对话×多智能体 |
| M4 | 过程奖励与对齐税 | 五级 PRM（含沙箱真实执行）、SFT 锚定批、奖励课程、轨迹回流、评测扩容 36 任务 + pass@k |
| M5 | 工具选择攻坚 | 动作示例×10 + 动作空间对比学习（工具选择 9×）、多步链式评测、v5 放大 23.4M、在线自我进化 evolve.py |
| M6 | 多轮上下文与安全 | (任务,历史,动作) 三元组 SFT、α 动态调度（对齐税再平衡）、语料 v5 + BPE 5000、grep/find 工具扩展、进化回归门禁 |
| M7 | 对齐税根治与多步穿透 | 多步链逐步入 GRPO（PRM 穿透）、参数级对比学习（L4 直训）、α 重锚定 0.7→0.2（QA 零征税）、轨迹批量积累 `--batch`、语料 v6 负边际收益实证 |

## 二、仓库结构

```
├── extracted/                  # 32 份职位说明（.doc 提取后的 Markdown）
├── antnest_team/               # 32 智能体团队（CrewAI）
│   ├── antnest_agents.py       #   32 个 Agent 定义（与职位一一对应）
│   ├── antnest_tasks.py        #   33 个任务编排
│   ├── main.py                 #   团队入口（结构校验/任务清单/运行）
│   └── outputs/                #   团队交付物（33~41 号：方案、周报、冲刺报告、过程详解）
└── antnest/                    # 产品本尊
    ├── antnest_llm/            # ── antNest LLM ──
    │   ├── model.py            #   TinyGPT（Decoder-only Transformer）
    │   ├── tokenizer.py        #   字符级 tokenizer（v1）
    │   ├── bpe.py              #   BPE 分词器（M2）
    │   ├── cleaner.py          #   语料清洗（去 .doc 噪声）
    │   ├── corpus.py           #   语料 v1~v6（v6 验证运行数据回灌的边际收益）
    │   ├── train.py            #   预训练入口
    │   ├── sft.py              #   监督微调（动作示例 + 多轮三元组 + --traj 轨迹回流）
    │   ├── grpo.py             #   GRPO RL（PRM + 多步穿透池 + 参数级对比 + α调度）
    │   ├── eval.py             #   独立评测（严格尺子 + pass@k + 多步链式）
    │   └── evolve.py           #   在线自我进化（含回归门禁）
    ├── antnest_harness/        # ── antNest Harness ──
    │   ├── agent.py            #   NestAgent：Agent Loop（```action JSON 协议）
    │   ├── tools.py            #   工具层 + DSec 沙箱（6 工具：含 grep/find 检索）
    │   ├── memory.py           #   短期/长期记忆
    │   ├── crew.py             #   NestCrew：planner→builder→reviewer
    │   ├── llm.py              #   LLM 抽象层（Mock / OpenAI 兼容 / 本地直连）
    │   ├── chat.py             #   本地模型多轮对话
    │   └── chat_crew.py        #   对话路由×Multi-Agent + 轨迹记录 + --batch 批量积累
    ├── evals/evalset.json      #   独立评测集（32 动作 + 4 多步 + 12 QA，与训练物理隔离）
    ├── corpus_extra.md         #   外部合规语料（公有领域典籍 + PSF 文本）
    ├── tests/                  #   质量门禁（M0~M7 共 97 项 pytest）
    └── artifacts/              #   训练产物（ckpt 被 .gitignore 排除，指标/配置/词表在库）
```

## 三、环境准备

```bash
# Python 3.10+，仅需 CPU
pip install torch pytest httpx crewai   # crewai 可选（只跑模型/Harness 无需）
```

> 仓库中的命令均假设 `cd antnest` 且 Python 可用；本仓库开发时使用 Python 3.12 + torch 2.13（CPU 版）。

## 四、快速开始（5 分钟跑通全链路）

```bash
cd antnest

# ① 预训练（语料 v4 = 32 JD + 团队交付物 + 产品源码 + 外部合规语料）
python -m antnest_llm.train --corpus v4 --tokenizer bpe --bpe_vocab 2600 \
       --n_embd 256 --n_head 8 --n_layer 8 --block 256 --steps 150 --prefix v4

# ② 监督微调（职位问答 + 产品知识 + 41 条动作协议示例 + 轨迹回流）
python -m antnest_llm.sft --steps 300 --base_prefix v4 --out_prefix sft6 \
       --traj artifacts/trajs.jsonl

# ③ 强化学习（GRPO：五级 PRM + 多步穿透池 + 参数级对比 + α 重锚定）
python -m antnest_llm.grpo --iters 30 --base_prefix sft6 --out_prefix grpo6 \
       --anchor_per_iter 4 --alpha_sft 0.7 --alpha_min 0.2

# ④ 独立评测（48 任务：32 动作 + 4 多步链式 + 12 QA；pass@3 + 自一致性）
python -m antnest_llm.eval --ckpt sft6,grpo6 --passk 3

# ⑤ 在线自我进化（新轨迹 → 增量 SFT → 评测择优晋升）
python -m antnest_llm.evolve --status
python -m antnest_llm.evolve --force

# ⑥ 对话 × 多智能体（自动加载最优 checkpoint：grpo > sft > 预训练）
PYTHONPATH=. python -m antnest_harness.chat_crew --message "统计团队交付物数量并写成报告"

# ⑥' 批量轨迹积累（M7：一条命令喂 8 个任务，轨迹连续入库）
PYTHONPATH=. python -m antnest_harness.chat_crew --batch artifacts/batch_tasks.txt

# ⑦ 质量门禁
python -m pytest tests/ -q
```

## 五、核心设计

### antNest LLM：三阶段训练 + 一把独立尺子

- **预训练**：TinyGPT（Decoder-only），warmup+余弦退火，验证集择优保存；语料从 v1（32 JD）滚到 v4（+交付物 +源码 +外部合规语料），"蚁巢用自己铸造自己"。
- **SFT**：`<|user|>/<|assistant|>` 模板，prompt 区间 -100 掩码；数据 = 职位问答 + 产品知识 + 动作协议示例 + **(任务, 历史, 动作) 多轮三元组**（上下文格式与多步评测的观察回填逐字一致）；`--traj artifacts/trajs.jsonl` 可把 Harness 真实工具调用轨迹回流为训练数据。
- **RL（GRPO + PRM + 锚定 + 对比学习 + α 调度 + 多步穿透）**：组采样归一化优势替代 critic；五级过程奖励
  `L1 格式 0.2 → L2 类型 0.2 → L3 选对工具 0.3 → L4 参数键 0.15 → L5 沙箱真实执行 0.15`；
  每步混入 SFT 锚定批（`loss = GRPO + α(t)·NLL`，**α 线性退火 0.7→0.2**：M7 重锚定配方，历代 GRPO 首次 QA 零征税）；
  奖励课程：L3-L5 前 10 迭代 2 折起步线性升满，解决严格奖励冷启动稀疏问题；
  **动作空间对比学习（两级）**：工具级（正例=期望工具 vs 负例=同格式误工具）
  + 参数级（正例=正确参数键 vs 负例=同工具错键名，M7 新增），pairwise margin
  损失（`+λ·L_ctr`），补上"GRPO 组内全错时无梯度"的死角；
  **多步穿透池**（M7）：多步链逐步展开为 GRPO 训练项（32 单步 + 21 逐步
  = 53 项），PRM 过程奖励逐级穿透到链式任务。
- **评测与训练分离**：`evals/evalset.json` 与训练任务池**措辞不同、语义等价、物理隔离**，`eval.py` 用同一把严格尺子横评所有 checkpoint（动作分 / pass@k / 自一致性 / 多步链式 / QA 命中率，共 48 任务）。
- **在线自我进化**：`evolve.py` 一键闭环——新轨迹检测 → 增量 SFT → 独立评测 → **回归门禁**（任一关键指标跌幅 >10% 拒绝晋升）→ 择优晋升 → 追加 `evolve_log.jsonl` 进化史（首次实证 evo1：overall 0.247 → 0.517）。

### antNest Harness：可插拔 LLM 的智能体外壳

- **Agent Loop**：系统提示（角色+工具说明）→ LLM → ` ```action ` JSON 解析 → 沙箱执行 → 观察回填，直至 `finish` 或上限。
- **DSec 沙箱**：所有路径限制在 `/workspace` 内（逃逸即拒），shell 白名单 + 超时；工具错误回传给 Agent 而非崩溃。
- **Multi-Agent**：NestCrew = 规划师 → 执行师 → 审查师，共享记忆。
- **LLM 抽象层**：`MockLLM`（离线测试）/ `OpenAICompatClient`（真实大模型，环境变量 `ANTNEST_LLM_API_BASE/_KEY/_MODEL`）/ `AntNestLLMClient`（本地自训模型直连，含 mini 模型脚手架解码兜底）。
- **轨迹回流**：`chat_crew.py` 的 `TrajRecorder` 记录每次真实工具调用 `(name, args, ok)` 到 `artifacts/trajs.jsonl`，成功轨迹经 `sft.py --traj` 蒸馏回模型——**使用即训练**；M7 起 `--batch` 模式一条命令批量积累轨迹（实测 8 任务 / 48 次工具调用 / 轨迹 3→11 条）。

## 六、已验证的演进结论

独立评测集（与训练物理隔离）上的里程碑对比（详表见 `antnest/artifacts/eval_compare.json`；M5 行 40 任务、M6 行 48 任务口径）：

| 模型 | 阶段 | 动作分 | pass@3 | QA | 多步 | 综合 | 结论 |
|---|---|---|---|---|---|---|---|
| sft4 | SFT(M3) | 0.000 | 0.000 | 0.167 | – | 0.056 | SFT 对训练措辞过拟合，OOD 不会动作 |
| grpo4 | +GRPO(M3) | 0.300 | 0.000 | 0.083 | – | 0.228 | RL 学会动作，但 QA 遗忘（对齐税） |
| grpo5 | +PRM+锚定(M4) | 0.287 | 0.000 | 0.167 | 0.300 | 0.247 | 锚定止血 QA；课程奖励修复冷启动 |
| **sft6** | +动作示例×10(M5*) | **0.713** | **0.583** | 0.167 | 0.200 | **0.531** | 数据扩容是第一生产力 |
| grpo6 | +对比学习(M5*) | 0.646 | 0.542 | 0.083 | **0.350** | 0.458 | RL 增益集中在过程能力（多步） |
| sft7 | 23M 大模型(M5*) | 0.525 | 0.333 | 0.125 | **0.658** | 0.392 | 规模换深度：多步 3.3× |
| **sft8** | +多轮三元组/v6(M6) | 0.559 | 0.344 | 0.167 | 0.438 | **0.452** | 工具扩容不打折：新工具任务零样本迁移 |
| grpo8 | +α 动态调度(M6) | **0.566** | **0.406** | 0.083 | **0.579** | 0.434 | α 释放策略空间：多步 +32%，QA 税待 M7 根治 |
| sft9 | 语料v6/v7(M7) | 0.506 | 0.312 | 0.167 | 0.325 | 0.414 | 运行数据全量回灌负边际：语料入库需筛选 |
| **grpo9** | +多步穿透/参数对比/α0.7(M7) | **0.591** | 0.344 | **0.167** | 0.342 | **0.475** | **对齐税根治：QA 与动作同涨，历代最优** |

\* M5 行为 40 任务口径，M6/M7 行为 48 任务口径（评测集新增 grep/find 8 条），横比见 40/41 号报告说明。

质量门禁：`pytest tests/` 97 项全绿（M0-M7），坏味道零容忍——每个 badcase 都有"定位→修复→回归测试"闭环记录（见冲刺报告 21/28/34/37/38/40/41 号）。

## 七、常见问题

- **`artifacts/*.pt` 在哪？** 模型权重超 GitHub 单文件限制被 `.gitignore` 排除；所有 `*_model_config.json / *_vocab.json / *_metrics.json` 均在库，按"快速开始"命令可完整复现每个 checkpoint。
- **想要更大模型？** 本库已含 25.2M 版（v6：`--corpus v5 --tokenizer bpe --bpe_vocab 5000 --n_embd 384 --n_head 6 --n_layer 12 --steps 300 --prefix v6`；v7 为语料 v6 版，M7 实证全量回灌运行数据会稀释动作分布，**推荐 v6/v5 语料**）；权重被 .gitignore 排除。
- **32 个智能体怎么协作的？** 见 `antnest_team/outputs/39_32智能体开发过程详解.md`——32 份 JD → 32 个 CrewAI Agent → 33 任务 DAG → 39 份交付物 → 代码落地的全记录。
- **接真实大模型？** 设置 `ANTNEST_LLM_API_BASE/_KEY/_MODEL` 后 Harness 全链路（Agent Loop / Crew / 对话路由）自动切换到 `OpenAICompatClient`。

## 八、License

- 代码：随仓库发布（Apache-2.0 精神，可自由使用与修改）
- 语料：`corpus_extra.md` 仅含公有领域文本（先秦典籍）与 PSF 许可证文本，无第三方版权风险
