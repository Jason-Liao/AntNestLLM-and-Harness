# 33 antNest 产品开发总体方案（汇总）

- 负责人：AI 产品经理（Agent 17），32 个职位团队联合交付

## 一、执行摘要
32 位 agent 完成了 antNest 计划 M0 里程碑：
**训练出 antNest LLM（mini）并交付可运行的 antNest Harness**，
全部产出经过 9/9 自动化测试验证，并实录一轮 badcase 进化闭环。

## 二、核心成果
1. **antNest LLM**（antnest/antnest_llm/）
   - 595K 参数 TinyGPT，自有 JD 语料 42,673 字符（词表 1,280）
   - 300 步预训练：train loss 7.34→2.44，val 7.15→4.08
   - 采样已呈现 JD 语域特征（ckpt.pt / sample.txt / metrics.json）
2. **antNest Harness**（antnest/antnest_harness/）
   - Agent Loop（JSON 动作协议、容错解析）
   - DSec 沙箱工具（路径根限制 + 命令白名单 + 超时）
   - 记忆与上下文管理（短期轮次 + 长期笔记 + 工具结果窗口压缩）
   - Subagent/Multi-Agent（planner→builder→reviewer）
   - LLM 可插拔（OpenAI 兼容 / Mock / 预留 antNest LLM 直连）
   - 离线演示真实完成统计任务并落盘报告
3. **质量与进化**：9/9 测试通过；"蚁巢 prompt 空编码"badcase 完成定位-修复-回归

## 三、依赖与风险
- 无外部 LLM API：Harness 以 Mock 验证，真实智能上限未释放（M1 首要解锁）
- 数据量：42K 字符远低于可用阈值，模型为管线验证件而非能力件
- 训练/评测数据同源：M1 必须分离

## 四、90 天行动计划（M1）
1. W1-2：接入真实 LLM 端点，Harness 跑通 3 类真实任务
2. W3-6：语料扩容 1000×（BPE、去重、配比），重训至可评估基线
3. W7-10：SFT + 评测集（14/15 号设计）建立能力基线
4. W11-13：Agent 任务 RL 环境对接（测试通过率为奖励）
