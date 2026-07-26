# ASR-Uncertainty Reranking 状态

最后更新：2026-07-26（Phase 0）

状态只使用：`TODO`、`IN_PROGRESS`、`WAITING_USER`、`RUNNING_REMOTE`、
`COMPLETED`、`FAILED`、`BLOCKED`。

| 阶段 | 任务 | 状态 | 当前 Commit | 远程状态 | 阻塞原因 | 下一步 |
|---|---|---|---|---|---|---|
| 0 | 工作区、环境、数据、模型、结果只读审计 | COMPLETED | 待本次提交 | 无 GPU/下载 | 无 | 固化 Phase 0 文档 |
| 0 | 协议与分阶段计划 | COMPLETED | 待本次提交 | 无 GPU/下载 | 无 | 等待用户确认三项协议/资源决定 |
| 0 | OEA audio prefix 协议锁 | COMPLETED | 待本次提交 | 未运行 | 无；冲突仍保留为 `[PAPER-CONFLICT]` | 主协议固定为文本 `query:`、audio-only no-prefix |
| 0 | Whisper/BGE/FiQA 首批下载批准 | COMPLETED | 待本次提交 | 未下载 | 无；D1–D4 已获用户批准 | 提交 pinned manifests/downloader；DATA-13B 后由 CPU 服务器执行 |
| 0 | D1–D4 固定清单与 CPU 下载 wrapper | COMPLETED | 待本次提交 | 未下载 | DATA-13B 运行时不 pull 共享仓库 | 35 项相关测试通过；push 后待 DATA-13B 结束再拉取执行 |
| 0/2 | SQuTR DATA-13B 内容门禁 | RUNNING_REMOTE | `cd29099` | CPU PID `8032`；`data13b_squtr_validation_20260726_141640` | 等待 CRC/schema/qrels/audio 最终退出码 | 只读监控；成功后提交小型证据 |
| 1 | Nemo base/+Cl 当前缓存实时只读复核 | WAITING_USER | 待后续提交 | 历史 LFS 审计 complete，尚未做 2026-07-26 live recheck | 需要远程 CPU 执行；全哈希可能超过 30 分钟 | DATA-13B 后执行独立资源审计 |
| 1 | Nemo inference-only checkpoint 与 model lock | TODO | N/A | 未执行 | 依赖 live resource audit | 生成小型锁并提交 |
| 1 | Nemo 5/25 OEA embedding smoke | TODO | N/A | 计划 1×RTX 4090 24GB | 依赖 model lock 和精确 GPU 命令批准 | 先提交精确 wrapper/config；预计 12–18 GiB |
| 1 | Nemo(+Cl) Clotho 表 2 T2A | TODO | N/A | 需要 1×GPU + CPU metrics | 依赖 smoke gate | 对照 21.57/47.16/60.36 |
| 2 | FiQA corpus/qrels 协议与文档构造锁 | TODO | N/A | DATA-13B 运行中 | 实际 schema/qrels 分布未完成 | DATA-13B 后固化 |
| 2 | B1/B2/B3 四条件一级召回 | TODO | N/A | 需要 GPU | 依赖数据/模型下载和 Phase 1 | 报告 nDCG/MRR/Recall/Oracle |
| 2 | Go/No-Go 审查 | BLOCKED | N/A | 未开始 | 依赖完整 Phase 2 结果 | 未达标不擅自改双路召回 |
| 3 | 1-best CE、固定融合、RRF、Gold 上限 | BLOCKED | N/A | 未开始 | 依赖 Go | 复用唯一 OEA Top-100 |
| 4 | 4-best、proxy posterior、不确定性 | BLOCKED | N/A | 未开始 | 依赖 Whisper 和 Phase 3 缓存 | CPU 单元测试后申请 GPU |
| 4 | FiQA train/dev TTS/speaker/noise 协议 | BLOCKED | N/A | 未下载/合成 | TTS、许可证、speaker/noise 隔离未批准 | 单独提交具体计划和预算 |
| 4 | Query/candidate gate 训练 | BLOCKED | N/A | 按用户要求未训练 | 依赖 TTS 和用户训练批准 | 只训练小 gate |
| 5 | FiQA 三 seed/消融/bootstrap/分析 | BLOCKED | N/A | 未开始 | 依赖 Phase 4 | 完整四条件正式评测 |
| 6 | NQ zero-shot | BLOCKED | N/A | 未开始 | 依赖 FiQA 六项门禁和新 GPU 批准 | 不在 NQ 调参 |
| 7 | 效率、表格、图和最终研究报告 | TODO | N/A | 未开始 | 依赖主实验 | 区分事实、推断、失败与局限 |

## 当前焦点

- 当前阶段：Phase 0 协议和资源范围已由用户确认，正在完成首批资源下载基础设施。
- 已完成：OEA/SQuTR/环境/模型证据审计；audio-only no-prefix 主协议锁定；D1–D4 下载批准；G1 固定为 1×RTX 4090 24GB 资源计划。
- 正在远程运行：原 OEA-5/SpeechXBT 工作流的 DATA-13B；该任务同时是新项目
  Phase 2 的数据门禁。
- 当前未执行：任何新下载、GPU 计算、TTS、模型或 gate 训练。
- 下一步：提交 D1–D4 精确清单和 CPU wrapper；先完成 DATA-13B 结果固化，再让共享服务器 pull 并执行下载。并行可在本地实现纯 CPU schema、缓存、normalization、N-best aggregation 和 metrics 单元测试。
