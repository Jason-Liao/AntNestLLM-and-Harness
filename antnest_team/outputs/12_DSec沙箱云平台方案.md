# 12 DSec Agent 沙箱云平台方案

- 负责人：Agent Infra 研发工程师（Agent 6）

## M0 已实现（antnest_harness/tools.py）
DSec 最小可用形态——进程级沙箱：
- 路径沙箱：SAFE_ROOT=/workspace，越界路径直接拒绝（测试 test_sandbox_blocks_escape 验证）
- 命令沙箱：shell 白名单（ls/cat/wc/python 等 12 个）+ 30s 超时
- 容错：工具异常转为文本观察值回传 Agent，不致进程崩溃

## M1+ 演进
- 容器级隔离：每 Agent 会话独立容器，cgroup 资源配额，缩小逃逸面
- 虚拟化网络：Agent 间通信不出物理边界的虚拟网络
- 临时存储：沙箱专供块设备/共享卷，会话结束即回收
