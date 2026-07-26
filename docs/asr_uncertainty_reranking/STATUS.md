# ASR-Uncertainty Reranking 状态

最后更新：2026-07-26（Phase 0）

状态只使用：`TODO`、`IN_PROGRESS`、`WAITING_USER`、`RUNNING_REMOTE`、
`COMPLETED`、`FAILED`、`BLOCKED`。

| 阶段 | 任务 | 状态 | 当前 Commit | 远程状态 | 阻塞原因 | 下一步 |
|---|---|---|---|---|---|---|
| 0 | 工作区、环境、数据、模型、结果只读审计 | COMPLETED | `711b6e8` | 无 GPU/下载 | 无 | 固化 Phase 0 文档 |
| 0 | 协议与分阶段计划 | COMPLETED | `711b6e8` | 无 GPU/下载 | 无 | 按已确认协议实施 |
| 0 | OEA audio prefix 协议锁 | COMPLETED | `711b6e8` | 未运行 | 无；冲突仍保留为 `[PAPER-CONFLICT]` | 主协议固定为文本 `query:`、audio-only no-prefix |
| 0 | Whisper/BGE/FiQA 首批下载批准 | COMPLETED | `6279b82` | 未下载 | 无；D1–D4 已获用户批准 | DATA-13B 后由 CPU 服务器执行 |
| 0 | D1–D4 固定清单与 CPU 下载 wrapper | COMPLETED | `6279b82` | 未下载 | DATA-13B 运行时不 pull 共享仓库 | 35 项相关测试通过；待 DATA-13B 结束再拉取执行 |
| 0 | CPU 核心算法与缓存契约 | COMPLETED | `8cc5984` | 未远程运行；本地 38 项相关测试通过 | 无 | 后续模型 runner 只能调用这些已测试定义 |
| 0/2 | SQuTR DATA-13B/13C 内容门禁 | RUNNING_REMOTE | `68aa615` | CPU host `bitahub-a20601801981030400524510` 正在运行 DATA-13C；PID `9533`；run `logs/data13c_squtr_content_recovery_20260726_155758`；启动时 reuse/validation/wrapper 均为 `PENDING` | 无新增阻塞；等待远程内容校验完成 | 收集三个退出码、probe cache/manifest 统计与 stdout/stderr；全为 0 后提交小型证据 |
| 1 | Nemo base/+Cl 当前缓存实时只读复核 | WAITING_USER | 待后续提交 | 历史 LFS 审计 complete，尚未做 2026-07-26 live recheck | 需要远程 CPU 执行；全哈希可能超过 30 分钟 | DATA-13B 后执行独立资源审计 |
| 1 | Nemo inference-only checkpoint 与 model lock | TODO | N/A | 未执行 | 依赖 live resource audit | 生成小型锁并提交 |
| 1 | Nemo 5/25 OEA embedding smoke | TODO | N/A | 计划 1×RTX 4090 24GB | 依赖 model lock 和精确 GPU 命令批准 | 先提交精确 wrapper/config；预计 12–18 GiB |
| 1 | Nemo(+Cl) Clotho 表 2 T2A | TODO | N/A | 需要 1×GPU + CPU metrics | 依赖 smoke gate | 对照 21.57/47.16/60.36 |
| 2 | FiQA corpus/qrels 协议与文档构造锁 | TODO | N/A | DATA-13B 运行中 | 实际 schema/qrels 分布未完成 | DATA-13B 后固化 |
| 2 | B1/B2/B3 四条件一级召回 | TODO | N/A | 需要 GPU | 依赖数据/模型下载和 Phase 1 | 报告 nDCG/MRR/Recall/Oracle |
| 2 | Go/No-Go 审查 | BLOCKED | N/A | 未开始 | 依赖完整 Phase 2 结果 | 未达标不擅自改双路召回 |
| 3 | 1-best CE、固定融合、RRF、Gold 上限 | BLOCKED | N/A | 未开始 | 依赖 Go | 复用唯一 OEA Top-100 |
| 4 | 4-best、proxy posterior、不确定性实验 | BLOCKED | `8cc5984` | 公式/特征/聚合 CPU 实现已完成；模型推理未开始 | 依赖 Whisper 下载、Phase 3 缓存和 GPU 批准 | 复用已测试 CPU 实现；后续仅补 Whisper adapter 与正式缓存 |
| 4 | FiQA train/dev TTS/speaker/noise 协议 | BLOCKED | N/A | 未下载/合成 | TTS、许可证、speaker/noise 隔离未批准 | 单独提交具体计划和预算 |
| 4 | Query/candidate gate 训练 | BLOCKED | N/A | 按用户要求未训练 | 依赖 TTS 和用户训练批准 | 只训练小 gate |
| 5 | FiQA 三 seed/消融/bootstrap/分析 | BLOCKED | N/A | 未开始 | 依赖 Phase 4 | 完整四条件正式评测 |
| 6 | NQ zero-shot | BLOCKED | N/A | 未开始 | 依赖 FiQA 六项门禁和新 GPU 批准 | 不在 NQ 调参 |
| 7 | 效率、表格、图和最终研究报告 | TODO | N/A | 未开始 | 依赖主实验 | 区分事实、推断、失败与局限 |

## 当前焦点

- 当前阶段：Phase 0 协议、资源范围、下载基础设施和通用 CPU 核心模块均已完成。
- 已完成：OEA/SQuTR/环境/模型证据审计；audio-only no-prefix 主协议锁定；D1–D4 下载批准和固定清单；G1 固定为 1×RTX 4090 24GB 资源计划；query-local normalization、4-best proxy-posterior 聚合、不确定性特征、检索/Oracle/WER/噪声指标和严格缓存 manifest。
- 最近远程结果：DATA-13B attempt 1 已退出；全量解压/CRC 成功，内容校验因
  FiQA 第 742 行空文档被本地严格策略拒绝。失败证据已保存；DATA-13C 已在执行
  commit `68aa615` 上启动恢复，PID `9533`，当前等待退出码和最终统计。
- 当前未执行：任何新下载、GPU 计算、TTS、模型或 gate 训练。
- 下一步：只读监控 `logs/data13c_squtr_content_recovery_20260726_155758`；reuse/validation/wrapper
  三个退出码全为 0 后固化 schema/manifest 统计，再启动已批准的 D1/D2–D4 CPU 下载。
  运行期间不让远程工作副本 pull。G1 仍须等 Nemo 实时审计、model lock 和精确 GPU
  命令提交后才启动。
