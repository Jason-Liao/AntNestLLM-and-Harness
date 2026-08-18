# 22 antNest Frontier 研究计划

- 负责人：Frontier（持续学习/自进化/新范式）研究员（Agent 23）

## 当前范式缺陷观察（来自 M0 实践）
- 模型无跨任务记忆：NestAgent 的 Memory 模块在模型外补足——
  记忆在 Harness 而非模型，是持续学习缺位的直接证据
- 自进化闭环已具雏形：badcase → 定位 → 修复 → 回归（本次真实走通），
  但进化发生在代码而非权重

## M1+ 候选方向
1. Agentic memory 写回训练：把 memory.md 转为继续训练数据
2. Self-distillation：Harness 执行轨迹 → 偏好数据 → 迭代后训练
3. 新架构探索：线性注意力长上下文（ctx 128 → 100K）的 scaling 实验
