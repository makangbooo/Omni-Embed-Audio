# ASR-Uncertainty Reranking 状态

最后更新：2026-07-26（Phase 0 confirmed；CPU 主实验框架完成）

状态只使用：`TODO`、`IN_PROGRESS`、`WAITING_USER`、`RUNNING_REMOTE`、
`COMPLETED`、`FAILED`、`BLOCKED`。

| 阶段 | 任务 | 状态 | 当前 Commit | 远程状态 | 阻塞原因 | 下一步 |
|---|---|---|---|---|---|---|
| 0 | 2026-07-26 主实验范围与远程状态复核 | COMPLETED | `9bf227c` | 用户已确认 Phase 0、D1–D4 与 G1 资源计划 | 无 | 按门禁进入 CPU 下载和数据审计 |
| 0 | 工作区、环境、数据、模型、结果只读审计 | COMPLETED | `711b6e8` | 无 GPU/下载 | 无 | 固化 Phase 0 文档 |
| 0 | 协议与分阶段计划 | COMPLETED | `711b6e8` | 无 GPU/下载 | 无 | 按已确认协议实施 |
| 0 | OEA audio prefix 协议锁 | COMPLETED | `711b6e8` | 未运行 | 无；冲突仍保留为 `[PAPER-CONFLICT]` | 主协议固定为文本 `query:`、audio-only no-prefix |
| 0 | Whisper/BGE/FiQA 首批下载批准 | COMPLETED | `6279b82` | 未下载 | 无；D1–D4 已获用户批准并再次确认有效 | 在独立 CPU 服务器执行 |
| 0 | D1–D4 固定清单与 CPU 下载 wrapper | COMPLETED | `6279b82` | 未下载 | 无；与 SQuTR 内容校验无数据依赖 | 35 项相关测试通过；拉取最新分支后执行 |
| 0 | CPU 核心算法与缓存契约 | COMPLETED | `8cc5984` | 未远程运行；本地 38 项相关测试通过 | 无 | 后续模型 runner 只能调用这些已测试定义 |
| 0 | B1–B7、QG、Ours、U1–U4、A1–A9 CPU 主实验框架 | COMPLETED | `8f28c55` | 本地 345 项全仓测试通过；无下载、GPU、模型推理或真实训练 | 无；合成 smoke 明确禁止写入研究结果 | 远程拉取后执行 D1–D4 下载、DATA-13C 恢复和 FiQA 数据审计 |
| 0 | B5/B6 主融合的 ASR 路由 | WAITING_USER | `8f28c55` | 配置暂定 `proxy_posterior`，1-best 路由保留为辅助诊断 | 用户原始定义未明确 B5/B6 使用 1-best 还是 4-best；会影响 A3 公平性 | 正式 FiQA dev/test 前确认；推荐 4-best 作为主路由 |
| 0 | D1–D4 远程下载执行 | WAITING_USER | `8f28c55` | 已批准、尚未下载；可与 SQuTR lock 审计在不同 CPU 服务器并行 | 需要用户启动远程命令 | 下载后返回两个 run 目录、exit code 与 manifest 状态 |
| 0/2 | SQuTR DATA-13B/13C 内容门禁 | WAITING_USER | `41efefa` | attempt 3 已结束且本机无相关进程；旧 lock 为 0-byte regular file，`lslocks`/`fuser`/`lsof` 未发现本机持有者，但 non-blocking `flock` 未取得锁 | 可能为另一台共享存储服务器持锁、遗留进程继承 FD，或共享文件系统锁状态；原探针把后续 `echo` 的退出码误记为 0 | 不删除 lock；在同目录用新 probe file 验证 `flock` 能力并正确捕获旧锁退出码，同时检查 `/proc/locks`/开放 FD |
| 1 | Nemo base/+Cl 当前缓存实时只读复核 | WAITING_USER | 待后续提交 | 历史 LFS 审计 complete，尚未做 2026-07-26 live recheck | 需要远程 CPU 执行；全哈希可能超过 30 分钟 | DATA-13B 后执行独立资源审计 |
| 1 | Nemo inference-only checkpoint 与 model lock | TODO | N/A | 未执行 | 依赖 live resource audit | 生成小型锁并提交 |
| 1 | Nemo 5/25 OEA embedding smoke | TODO | N/A | 计划 1×RTX 4090 24GB | 依赖 model lock 和精确 GPU 命令批准 | 先提交精确 wrapper/config；预计 12–18 GiB |
| 1 | Nemo(+Cl) Clotho 表 2 T2A | TODO | N/A | 需要 1×GPU + CPU metrics | 依赖 smoke gate | 对照 21.57/47.16/60.36 |
| 2 | FiQA corpus/qrels 协议与文档构造锁 | TODO | `8f28c55` | 严格审计器已实现；真实数据尚未执行 | 等待 D1 与 DATA-13C final manifest | 运行 `audit_asrur_fiqa_data.py` 并固化 count/hash/leakage |
| 2 | B1/B2/B3 四条件一级召回 | TODO | N/A | 需要 GPU | 依赖数据/模型下载和 Phase 1 | 报告 nDCG/MRR/Recall/Oracle |
| 2 | Go/No-Go 审查 | BLOCKED | N/A | 未开始 | 依赖完整 Phase 2 结果 | 未达标不擅自改双路召回 |
| 3 | 1-best CE、固定融合、RRF、Gold 上限 | BLOCKED | N/A | 未开始 | 依赖 Go | 复用唯一 OEA Top-100 |
| 4 | 4-best、proxy posterior、不确定性实验 | BLOCKED | `8f28c55` | 缓存装配、聚合、特征、dev 选择、正式评测实现完成；模型推理未开始 | 依赖 Whisper 下载、Phase 3 缓存和 GPU 批准 | 补模型 adapter 后生成正式缓存 |
| 4 | FiQA train/dev TTS/speaker/noise 协议 | BLOCKED | N/A | 未下载/合成 | TTS、许可证、speaker/noise 隔离未批准 | 单独提交具体计划和预算 |
| 4 | Query/candidate gate 训练 | BLOCKED | `8f28c55` | 三 seed、小 MLP、multi-positive listwise、zero-positive 审计已实现；仅合成测试 | 依赖 TTS 协议和用户真实训练批准 | 只训练小 gate；上游模型保持冻结 |
| 5 | FiQA 三 seed/消融/bootstrap/分析 | BLOCKED | `8f28c55` | A1–A9、paired bootstrap、噪声/gate/统一 CSV 已实现；无真实数值 | 依赖 Phase 4 正式缓存 | 完整四条件正式评测 |
| 6 | NQ zero-shot | BLOCKED | N/A | 未开始 | 依赖 FiQA 六项门禁和新 GPU 批准 | 不在 NQ 调参 |
| 7 | 效率、表格、图和最终研究报告 | TODO | N/A | 未开始 | 依赖主实验 | 区分事实、推断、失败与局限 |

## 当前焦点

- 当前阶段：Phase 0 已由用户确认；无模型 CPU 主实验框架已经完成并通过全仓
  345 项测试。真实模型推理、TTS 与 gate 训练仍未开始。
- 当前主实验：FiQA 四种 SQuTR 声学条件上的 OEA Top-100 + Whisper 4-best
  proxy-posterior + BGE Cross-Encoder + ASR 不确定性感知候选级动态门控；
  OEA-5/SQuTR 数据门禁与 Clotho 正确性检查只是前置条件，不是最终创新实验。
- 已完成：OEA/SQuTR/环境/模型证据审计；audio-only no-prefix 主协议锁定；D1–D4
  下载批准和固定清单；G1 固定为 1×RTX 4090 24GB；FiQA/SQuTR 泄漏审计、
  全 corpus dense ranking、冻结 Top-100、B4–B7、U2–U4、query/candidate gate、
  A1–A9、三 seed、bootstrap、噪声/gate/CSV 汇总均已编码和合成验证。
- 最近远程结果：DATA-13C attempt 3 使用执行 commit `db7356c` 启动，但 PID
  `11079` 已退出。run `data13c_squtr_content_recovery_20260726_161849` 的
  wrapper exit=`20`，reuse/validation exit 文件、v2 cache 和 final manifest
  均未生成。后续在 commit `41efefa` 上检查确认本机无相关进程；
  `.data13c_content_recovery.lock` 是 0-byte regular file，且
  `lslocks`/`fuser`/`lsof` 均未显示本机持有者，但 non-blocking `flock` 仍未取得
  锁。原命令末尾显示的 `LOCK_PROBE_EXIT=0` 是后续 `echo` 的退出码，不是
  `flock` 的退出码。必须继续做同目录能力探针和跨服务器持有者排查，不得删除或
  绕过 lock。
- 当前未执行：D1–D4 实际下载、任何 GPU 计算、TTS、模型推理或真实 gate 训练。
- 下一步：两台 CPU 服务器可以并行执行：(1) D1 与 D2–D4 固定资源下载；
  (2) SQuTR lock-owner/availability 只读审计。不得删除 lock。DATA-13C 成功且
  D1 完成后运行 FiQA/SQuTR 一致性审计。G1 仍须等 Nemo 实时审计、model lock
  和精确 GPU 命令再次提交后才启动。
