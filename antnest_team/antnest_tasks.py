# -*- coding: utf-8 -*-
"""
antNest 团队任务流程 —— 让 32 位 Agent 协作开发 antNest 产品线：
antNest LLM（蚁巢大模型） + antNest Harness（智能体外壳）。

流程（顺序执行，通过 context 传递上下游产出）：
  P0 产品定义      → AI 产品经理制定双产品线战略
  P1 基础保障      → IDC / 采购 / 财务 / 法务 给出承载、供给、预算与合规方案
  P2 算力与系统底座 → 超算集群、算子/通信/编译器、训推框架、分布式存储、
                      平台运维、IT 基础设施、Agent Infra（DSec 沙箱云）
  P3 数据燃料      → 预训练数据管线 + Code/通用Agent/专业领域/创作/情感 五路数据与评测
  P4 模型研发      → 预训练 / 后训练 / 多模态 / Frontier / 深度学习研发
  P5 Harness 研发  → Agent Harness 团队给出 antNest Harness 总体设计
  P6 产品工程      → AI 搜索、服务端、前端客户端、测试开发落地产品
  P7 组织保障      → HR、管培生、行政、跨界人才支撑团队运转
  P8 汇总交付      → AI 产品经理汇总为《antNest 产品开发总体方案》
"""
import os
from crewai import Task

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _out(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def build_tasks(agents: dict) -> list:
    """基于 32 个 Agent 构建产品开发任务流。"""
    t = {}

    # ── P0 产品定义 ──────────────────────────────────────────────
    t[1] = Task(
        description=(
            "antNest（蚁巢）模型计划启动：目标是训练出 antNest LLM，并打造 antNest Harness。"
            "请你作为 AI 产品经理，制定 antNest 双产品线的整体战略与路线图：明确两条产品线"
            "各自的定位、目标用户、核心场景、里程碑与优先级，并说明 LLM 与 Harness 如何"
            "相互驱动、共同进化。"
        ),
        expected_output="一份《antNest 产品战略与路线图》，含产品定位、双产品线里程碑、协同机制。",
        agent=agents[17],
        output_file=_out("01_产品战略与路线图.md"),
    )

    # ── P1 基础保障 ─────────────────────────────────────────────
    t[2] = Task(
        description=(
            "基于产品战略，规划支撑 antNest 训练与推理的数据中心方案：容量与功率规划、"
            "风冷/液冷技术路线、供配电与制冷设计要点、可用性与能效指标，以及从 MW 级到 "
            "GW 级的扩展路径。"
        ),
        expected_output="《antNest 数据中心承载规划》，含容量、供电制冷、能效与扩展方案。",
        agent=agents[16],
        context=[t[1]],
        output_file=_out("02_数据中心承载规划.md"),
    )
    t[3] = Task(
        description=(
            "依据产品战略与数据中心规划，制定算力采购方案：GPU/服务器等核心硬件的寻源与"
            "供应商策略、云服务与 SaaS 采购策略、IDC 资源采购要点，以及采购-资产全生命周期"
            "管理机制。"
        ),
        expected_output="《antNest 算力与资源采购方案》，含硬件/云/IDC 采购策略与资产管理。",
        agent=agents[31],
        context=[t[1], t[2]],
        output_file=_out("03_算力与资源采购方案.md"),
    )
    t[4] = Task(
        description=(
            "为 antNest 计划编制财务方案：训练与推理算力的成本模型、全面预算管理体系、"
            "资本运作与投融资要点，以及合规风控体系，让每一份投入可度量、可回溯。"
        ),
        expected_output="《antNest 财务与预算方案》，含算力成本模型、预算与风控体系。",
        agent=agents[30],
        context=[t[1], t[3]],
        output_file=_out("04_财务与预算方案.md"),
    )
    t[5] = Task(
        description=(
            "为 antNest 产品线建立法律与合规框架：训练数据合规（版权、隐私、跨境）、模型"
            "服务合规（生成内容、备案）、开源与知识产权策略、用户协议要点，识别主要法律"
            "风险并给出应对措施。"
        ),
        expected_output="《antNest 合规与法律风险框架》，含数据合规、模型合规与 IP 策略。",
        agent=agents[29],
        context=[t[1]],
        output_file=_out("05_合规与法律风险框架.md"),
    )

    # ── P2 算力与系统底座 ───────────────────────────────────────
    t[6] = Task(
        description=(
            "设计 antNest 训练超算集群架构：集群规模与拓扑（万卡→数十万卡）、异构算力调度"
            "（训推一体）、高性能网络（RDMA/RoCEv2/InfiniBand）、故障自愈与断点续训机制，"
            "以及集群与模型的协同设计要点。"
        ),
        expected_output="《antNest 超算集群架构方案》，含拓扑、调度、网络与容灾设计。",
        agent=agents[10],
        context=[t[1], t[2], t[3]],
        output_file=_out("06_超算集群架构方案.md"),
    )
    t[7] = Task(
        description=(
            "面向 antNest LLM 的训练与推理负载，给出高性能算子与通信方案：GEMM/Attention "
            "算子优化路线、集合通信（NCCL/DeepEP 等）调优要点、编译器/DSL（TileLang/Triton）"
            "能力建设，以及与 GPU/NPU 硬件极限的差距分析。"
        ),
        expected_output="《antNest 高性能算子/通信/编译器方案》，含优化路线与性能目标。",
        agent=agents[11],
        context=[t[6]],
        output_file=_out("07_高性能算子通信编译器方案.md"),
    )
    t[8] = Task(
        description=(
            "设计 antNest 大模型训练/推理框架方案：模型并行与长上下文、MoE 与低精度训推、"
            "RL 训练系统（异步 RL、Agent RL）、多模态训练支持，以及大规模推理服务的 KV "
            "Cache 缓存与负载均衡策略。"
        ),
        expected_output="《antNest 训练推理框架方案》，含并行策略、RL 系统与推理服务设计。",
        agent=agents[12],
        context=[t[6], t[7]],
        output_file=_out("08_训练推理框架方案.md"),
    )
    t[9] = Task(
        description=(
            "设计 antNest 高性能分布式存储方案：面向推理的 KVCache 存储系统（毫秒级延迟、"
            "上亿级 IOPS）、面向训练的分布式文件系统/对象存储、数据湖与共享存储层，以及"
            "关键一致性机制选型。"
        ),
        expected_output="《antNest 分布式存储方案》，含 KVCache 存储、训练数据存储架构。",
        agent=agents[13],
        context=[t[6], t[8]],
        output_file=_out("09_分布式存储方案.md"),
    )
    t[10] = Task(
        description=(
            "为 antNest 训推平台制定运维保障方案：全栈可观测体系（GPU 利用率、NCCL 抖动、"
            "checkpoint 耗时）、故障自愈与自动上下线、快与稳的平衡策略，以及 CMDB/工单等"
            "运维平台建设要点。"
        ),
        expected_output="《antNest 平台运维与可观测方案》，含监控、自愈与平台建设。",
        agent=agents[14],
        context=[t[6], t[8]],
        output_file=_out("10_平台运维与可观测方案.md"),
    )
    t[11] = Task(
        description=(
            "给出 antNest 系统硬件与网络基线：GPU/网卡/服务器选型建议、固件与配置管理、"
            "基准压测框架（fio/iperf3/gpu-burn）、跨地域网络互联与 RDMA 运维要点，以及"
            "研发办公 IT 保障（桌面、账号、会议、网络）方案。"
        ),
        expected_output="《antNest 硬件网络基线与 IT 保障方案》，含选型、压测与办公 IT。",
        agent=agents[15],
        context=[t[6], t[2]],
        output_file=_out("11_硬件网络基线与IT保障.md"),
    )
    t[12] = Task(
        description=(
            "设计 DSec——antNest 为 Agent 量身定制的沙箱云平台：大规模虚拟机/容器隔离与"
            "资源管控、沙箱网络与临时存储、混合云架构与管控面，以及无监督全自动 Agent 的"
            "安全边界设计，为 Harness 与模型训练提供运行环境。"
        ),
        expected_output="《DSec Agent 沙箱云平台方案》，含隔离、网络、存储与安全设计。",
        agent=agents[6],
        context=[t[6], t[8]],
        output_file=_out("12_DSec沙箱云平台方案.md"),
    )

    # ── P3 数据燃料 ─────────────────────────────────────────────
    t[13] = Task(
        description=(
            "设计 antNest LLM 预训练数据体系：全网数据选取与采集策略、语料清洗与去重"
            "（MinHash/向量去重）管线、数据配比与质量治理、多模态数据管理，以及数据基建"
            "（KV/消息队列/数据湖）选型。"
        ),
        expected_output="《antNest 预训练数据管线方案》，含采集、清洗、配比与数据基建。",
        agent=agents[3],
        context=[t[1], t[8]],
        output_file=_out("13_预训练数据管线方案.md"),
    )
    t[14] = Task(
        description=(
            "为 antNest LLM 的代码能力设计 RL 训练环境与评测体系：把真实软件工程场景转化为"
            "可训练环境、设计奖励信号、构建前端/后端/移动端/安全等专项评测任务，建立能力"
            "短板的定位与数据补全闭环。"
        ),
        expected_output="《Code Agent 训练环境与评测体系》，含 RL 环境、奖励与评测设计。",
        agent=agents[18],
        context=[t[1], t[13]],
        output_file=_out("14_CodeAgent环境与评测.md"),
    )
    t[15] = Task(
        description=(
            "设计通用 Agent（办公/生活/搜索）场景的数据与评测体系：端到端任务完成度与过程"
            "行为质量的评测维度、自动化评测与归因方法、『能用』vs『好用』的区分标准，以及"
            "高质量数据生产管线设计。"
        ),
        expected_output="《通用 Agent 评测与数据体系》，含评测维度、归因与数据管线。",
        agent=agents[19],
        context=[t[1], t[13]],
        output_file=_out("15_通用Agent评测与数据.md"),
    )
    t[16] = Task(
        description=(
            "针对小语种、医学、法律等专业领域，设计模型能力评测与数据方案：专业评估维度"
            "搭建、自动化评估方式、优秀回答标准定义，以及兼具实用与审美价值的数据制作"
            "流程。"
        ),
        expected_output="《专业领域评测与数据方案》，含垂类评测维度与数据标准。",
        agent=agents[20],
        context=[t[1], t[13]],
        output_file=_out("16_专业领域评测与数据.md"),
    )
    t[17] = Task(
        description=(
            "为 antNest 模型的创作能力设计评测与数据方案：文学写作（小说/诗歌/散文）与功能"
            "写作（报告/公文/文案）的理想输出标准、审美维度拆解、质量评估体系，以及驱动"
            "写作能力进化的数据生产策略。"
        ),
        expected_output="《AI 创作能力评测与数据方案》，含文体标准与审美维度。",
        agent=agents[21],
        context=[t[1], t[13]],
        output_file=_out("17_AI创作评测与数据.md"),
    )
    t[18] = Task(
        description=(
            "为 antNest 模型的角色扮演与情感陪伴能力设计评测与优化方案：真实感与沉浸度的"
            "评估维度、典型 Badcase 归因框架、人机情感交互的产品形态建议，以及对应的数据"
            "生产与迭代策略。"
        ),
        expected_output="《情感智能评测与优化方案》，含体验维度、归因与数据策略。",
        agent=agents[22],
        context=[t[1], t[13]],
        output_file=_out("18_情感智能评测与优化.md"),
    )

    # ── P4 模型研发 ─────────────────────────────────────────────
    t[19] = Task(
        description=(
            "制定 antNest LLM 预训练技术方案：模型结构设计（软硬件协同）、优化器与训练动力"
            "学、scaling law 规划、训练加速策略，以及基于预训练数据管线的数据配比与治理"
            "策略。"
        ),
        expected_output="《antNest LLM 预训练技术方案》，含架构、优化器、scaling law。",
        agent=agents[24],
        context=[t[13], t[6], t[8]],
        output_file=_out("19_预训练技术方案.md"),
    )
    t[20] = Task(
        description=(
            "制定 antNest LLM 后训练方案：RLHF/RLVR/PPO/GRPO 等算法选型与迭代计划、高质量"
            "后训练数据集构建、自动化清洗/评测/合成管线，以及覆盖写作、问答、Agent 场景的"
            "评测体系。"
        ),
        expected_output="《antNest LLM 后训练方案》，含 RL 算法、数据与评测体系。",
        agent=agents[25],
        context=[t[19], t[14], t[15], t[16], t[17], t[18]],
        output_file=_out("20_后训练方案.md"),
    )
    t[21] = Task(
        description=(
            "制定 antNest 多模态理解方案：视觉编码器选型与优化、多模态预训练与后训练"
            "（SFT/RL/OPD）策略、图文/视频数据体系建设，以及 GUI、文档解析、多模态搜索等"
            "Agent 场景的落地路径。"
        ),
        expected_output="《antNest 多模态理解方案》，含编码器、训练策略与场景落地。",
        agent=agents[26],
        context=[t[19], t[13]],
        output_file=_out("21_多模态理解方案.md"),
    )
    t[22] = Task(
        description=(
            "围绕持续学习与自进化新范式，提出 antNest 的 Frontier 研究计划：当前范式的关键"
            "缺陷、持续学习/自进化/新架构的候选方向、实验设计与验证路径，以及对 antNest "
            "LLM 长期演进的路线建议。"
        ),
        expected_output="《antNest Frontier 研究计划》，含新范式候选与实验路线。",
        agent=agents[23],
        context=[t[19]],
        output_file=_out("22_Frontier研究计划.md"),
    )
    t[23] = Task(
        description=(
            "以『算法+系统』双重视角，对预训练与后训练方案进行精度-性能联合审视：训练与"
            "推理部署的平衡点、算子与框架层面的加速建议、部署形态与成本优化，输出工程"
            "落地视角的修订意见。"
        ),
        expected_output="《算法-系统联合优化意见》，含精度性能平衡与部署建议。",
        agent=agents[2],
        context=[t[19], t[20], t[7], t[8]],
        output_file=_out("23_算法系统联合优化.md"),
    )

    # ── P5 Harness 研发 ─────────────────────────────────────────
    t[24] = Task(
        description=(
            "这是 antNest 计划的核心交付之一。请设计 antNest Harness 总体方案：技术架构与"
            "选型（Agent Loop、Tool Use、上下文管理、长期记忆、Subagent/Multi-Agent、自进化"
            "Agent、超长程任务）、与 antNest LLM 的深度适配与共同进化机制、基准测试与评测"
            "方法、基于真实任务与用户反馈的迭代闭环，并给出 MVP 范围与实施计划。"
        ),
        expected_output="《antNest Harness 总体设计方案》，含架构、模型适配、评测与 MVP 计划。",
        agent=agents[5],
        context=[t[1], t[12], t[19], t[20], t[14], t[15]],
        output_file=_out("24_antNestHarness总体设计.md"),
    )

    # ── P6 产品工程 ─────────────────────────────────────────────
    t[25] = Task(
        description=(
            "设计 antNest AI 搜索基础设施：LLM 原生检索系统架构、query 理解/召回/排序核心"
            "算法、多语言多模态检索能力、搜索质量评估体系，以及作为 Harness 工具的接入"
            "方式。"
        ),
        expected_output="《antNest AI 搜索方案》，含架构、算法与质量评估。",
        agent=agents[4],
        context=[t[24], t[19]],
        output_file=_out("25_AI搜索方案.md"),
    )
    t[26] = Task(
        description=(
            "设计 antNest 线上核心服务架构：面向数千万日活的大模型应用与 API 服务、大模型"
            "研究中台（可观测/可视化）、数据仓库与数据管道，以及服务的可靠性、可扩展性与"
            "可观测性方案。"
        ),
        expected_output="《antNest 服务端架构方案》，含 API 服务、中台与数据仓库。",
        agent=agents[1],
        context=[t[24], t[25], t[9]],
        output_file=_out("26_服务端架构方案.md"),
    )
    t[27] = Task(
        description=(
            "设计 antNest 产品的前端与客户端方案：网页端/APP 端信息架构与核心交互、Agent "
            "协作与任务交付的新交互范式、性能与体验优化要点，以及与 Harness 能力对应的"
            "界面呈现。"
        ),
        expected_output="《antNest 前端客户端方案》，含信息架构与 Agent 交互范式。",
        agent=agents[7],
        context=[t[24], t[26]],
        output_file=_out("27_前端客户端方案.md"),
    )
    t[28] = Task(
        description=(
            "为 antNest 产品线制定质量保障方案：模型输出的自动化评测门禁、服务端与客户端"
            "测试体系、Agent 端到端任务质量评估、badcase 分析闭环，以及上线前质量 checklist。"
        ),
        expected_output="《antNest 质量保障方案》，含评测门禁与测试体系。",
        agent=agents[8],
        context=[t[24], t[25], t[26], t[27]],
        output_file=_out("28_质量保障方案.md"),
    )

    # ── P7 组织保障 ─────────────────────────────────────────────
    t[29] = Task(
        description=(
            "为 antNest 计划制定人才战略：32 个职位的团队编制与关键人才画像、海内外高潜"
            "人才寻访策略、人才资源池建设，以及支撑 AGI 探索的组织文化与保留机制。"
        ),
        expected_output="《antNest 人才战略与组织方案》，含编制、画像与寻访策略。",
        agent=agents[27],
        context=[t[1]],
        output_file=_out("29_人才战略与组织.md"),
    )
    t[30] = Task(
        description=(
            "以管培生视角梳理 antNest 计划的业务全貌：各阶段关键交付之间的业务逻辑链条、"
            "新人快速上手的学习路径、跨团队协作中的信息断层与改进建议。"
        ),
        expected_output="《antNest 业务全貌与轮岗学习路径》，含流程梳理与改进建议。",
        agent=agents[28],
        context=[t[1], t[24]],
        output_file=_out("30_业务全貌与学习路径.md"),
    )
    t[31] = Task(
        description=(
            "为 antNest 团队制定行政保障方案：研发办公环境与设备保障、会议与协作空间、"
            "员工体验与文化建设，以及用 AI 工具提升行政效率的落地做法。"
        ),
        expected_output="《antNest 行政保障方案》，含环境、体验与 AI 赋能做法。",
        agent=agents[32],
        context=[t[29]],
        output_file=_out("31_行政保障方案.md"),
    )
    t[32] = Task(
        description=(
            "以跨界视角审视 antNest 计划：从其他学科/行业可借鉴的方法论、当前方案中的"
            "思维盲区、非常规的机会点，给出三条以上具体可执行的跨界创新建议。"
        ),
        expected_output="《antNest 跨界创新建议》，含方法论借鉴与非常规机会点。",
        agent=agents[9],
        context=[t[1], t[24]],
        output_file=_out("32_跨界创新建议.md"),
    )

    # ── P8 汇总交付 ─────────────────────────────────────────────
    t[33] = Task(
        description=(
            "汇总全部 32 个职位团队的产出，编制《antNest 产品开发总体方案》：执行摘要、"
            "产品战略、基础设施、数据、模型、Harness、产品工程、组织保障各章节的关键结论"
            "与行动项，标注风险与依赖，并给出下一步 90 天行动计划。"
        ),
        expected_output="《antNest 产品开发总体方案》，整合各团队产出与 90 天行动计划。",
        agent=agents[17],
        context=[t[i] for i in range(1, 33)],
        output_file=_out("33_antNest产品开发总体方案.md"),
    )

    return [t[i] for i in range(1, 34)]
