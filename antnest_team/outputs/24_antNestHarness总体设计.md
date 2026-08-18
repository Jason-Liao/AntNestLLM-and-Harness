# 24 antNest Harness 总体设计方案（核心交付）

- 负责人：Agent Harness 团队（Agent 5）；沙箱由 Agent 6 提供

## 1. 设计原则
研究-工程-产品一体：所有概念均可运行验证；模型无关（LLM 可插拔）；
安全默认（工具全部沙箱化）。

## 2. 架构

```
用户任务
   │
   ▼
NestAgent（agent.py，Agent Loop）
   │  系统提示 = 角色/目标/背景 + 工具说明 + 动作协议
   │
   ├──► LLM 抽象层（llm.py）
   │      ├─ OpenAICompatClient（任意兼容端点，env: ANTNEST_LLM_*）
   │      ├─ MockLLM（离线自检/测试）
   │      └─ [预留] 本地 antNest LLM 直连（模型-Harness 共同进化接口）
   │
   ├──► ToolRegistry（tools.py，DSec 沙箱）
   │      list_dir / read_file / write_file / shell
   │      路径根限制 + 命令白名单 + 超时 + 容错回传
   │
   └──► Memory（memory.py）
          短期：轮次级对话；长期：memory.md 跨任务笔记
          上下文管理：工具结果窗口化压缩（保 system + 最近 N 轮）

NestCrew（crew.py）＝ Subagent / Multi-Agent
   planner（拆解）→ builder（工具执行）→ reviewer（校验）
```

## 3. 动作协议
LLM 以 ```action JSON 块输出 `tool` / `finish`；解析容错（正则回退），
格式错误作为观察值回传自动纠正。

## 4. M0 验证结果
- 离线演示：MockLLM 驱动 + 真实沙箱工具，完成"统计职位文件并生成报告"
  （artifacts/harness_demo_report.md），Loop 收敛于 finish ✔
- 测试：9/9 通过（Loop/沙箱/记忆/多智能体全覆盖）✔
- badcase 闭环：采样 prompt 空编码 bug → 定位 → 修复 → 回归 ✔

## 5. M1+ 路线
真实 LLM 端点接入 → antNest LLM 本地直连 → 评测集驱动迭代（15 号）→
Skills/MCP 扩展 → 长程任务（checkpoint 式任务状态）。
