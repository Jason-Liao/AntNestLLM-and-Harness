# antNest 全量测试报告

> 生成时间：2026-08-19 | 环境：Python 3.12 + torch 2.13（CPU）
> 命令：`python -m pytest tests/ -v --junitxml=artifacts/test_results.xml`

## 总览

| 指标 | 结果 |
|---|---|
| 总测试数 | **97** |
| 通过 | **97（100%）** |
| 失败 / 错误 | 0 |
| 总耗时 | ~26s（CPU） |
| 最慢单测 | `test_m1.py::test_antnest_llm_direct_connection` 21.1s（加载 25.2M checkpoint 实跑生成） |

## 分模块明细

| 模块 | 数量 | 覆盖内容 |
|---|---|---|
| test_m1.py | 5 | 语料 v2、SFT 示例与损失掩码、生成长提示、LLM 直连 Harness、checkpoint-词表一致性 |
| test_m2.py | 5 | 语料清洗去 .doc 噪声、BPE 往返与压缩比、tokenizer 分发、v3 一致性生成、多轮对话 |
| test_m3.py | 13 | GRPO 奖励（全分/部分/塑形/类型错）、任务池完整性、评测集与训练物理隔离、严格评分、QA 命中、路由全覆盖、Mock LLM 的 Crew 闭环、语料 v4 外部合规 |
| test_m4.py | 13 | 五级 PRM（L3 工具选择/L4 参数键/L5 沙箱真实执行）、沙箱直通、工具参数表、锚定损失、轨迹回流加载、轨迹记录器、评测集扩容 36 任务、pass@k |
| test_m5.py | 10 | 动作示例扩容、任务池扩容、对比对格式、对比损失梯度、多步链式评测、进化 skip/日志、进化 CLI、sft6/grpo6 产物一致 |
| test_m6.py | 25 | 多轮三元组（存在/格式对齐/隔离/下一步正确）、α 调度端点与单调、语料 v5 超集+运行数据、grep/find 沙箱与逃逸拦截、6 工具注册、新工具 SFT/GRPO/PRM/评测覆盖、评测措辞防泄漏、回归门禁（通过/拒绝/None 容错）、v6 词表 5000、grpo8 α 逐迭代记录 |
| test_m7.py | 17 | 多步穿透池（展开/格式对齐/finish 步/隔离）、参数级负例键错误、参数对 35 组、参数对比梯度、L4 仍计分、α 重锚定边界、grpo9 α 实验记录、--batch 任务读取与路由、批量后轨迹增长、语料 v6 超集+运行数据、sft9/grpo9 产物与奖励史 |
| test_products.py | 9 | 分词器往返、模型前向形状、训练损失下降、沙箱逃逸拦截、工具写读、Agent Loop（finish/工具后 finish）、多智能体协作、Harness 产物存在 |

## 质量门禁设计要点

1. **评测集永不进训练梯度**——`test_evalset_separation_from_training` 等多个测试盯防措辞泄漏；
2. **回归门禁**——`test_regression_gate_*` 三态验证（改进放行 / 单指标暴跌拒绝 / None 容错），已两次实证拦截 evo3/evo4；
3. **沙箱安全**——`test_grep_tool_escape_blocked`、`test_sandbox_blocks_escape` 验证 DSec 路径逃逸拒绝；
4. **训练-评测格式逐字对齐**——`test_multiturn_format_matches_eval_loop`、`test_multistep_pool_ctx_format_matches_eval` 保证训练分布与评测回填一致；
5. **α 调度可审计**——`test_grpo9_alpha_experiment_logged` 验证每个迭代的 α 写入训练 history。

## 复现

```bash
cd antnest
pip install torch pytest httpx
python -m pytest tests/ -v                          # 全量 97 项
python -m pytest tests/ --junitxml=artifacts/test_results.xml   # JUnit XML（CI 用）
```

> 机器可读结果见同目录 `test_results.xml`（JUnit 标准格式，可直接接入 GitHub Actions / CI）。
