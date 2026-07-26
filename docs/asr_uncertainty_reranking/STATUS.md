# ASR-Uncertainty Reranking 状态

最后更新：2026-07-27（Nemo Phase 1 metadata gate 修复）

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
| 0 | B5/B6 主融合的 ASR 路由 | COMPLETED | `b9a81ef` | `[INFERRED][USER-CONFIRMED 2026-07-26]` 主路线固定为 4-best `proxy_posterior`；1-best 只作辅助诊断 | 无 | 正式结果不得根据 test/NQ 表现切换路线 |
| 0 | D1 FiQA 固定资源下载 | COMPLETED | `41efefa` | run=`asrur_d1_fiqa_download_20260726_200414`；download/wrapper exit=`0/0`，manifest=`complete`，5/5 selected files 完成，目标目录约 47 MiB | 无 | DATA-13D 生成目标 manifest 后在同一 run 执行一致性审计 |
| 0 | 模型存储迁移与旧缓存清理 | COMPLETED | `aac088d` | run=`model_cache_migration_v3_20260726_214834`；exit=`0`；OEA 254 files、apparent/allocated bytes 前后完全一致；旧 `/home/jg525/model_cache` 已按用户明确授权删除 | 无 | 后续统一使用 `/home/jg525/models`；历史审计中的旧路径保持原样 |
| 0 | D2–D4 Whisper/BGE 固定资源下载 | RUNNING_REMOTE | `aac088d` | 旧顺序 run 已按用户要求以 `143/143` 停止；三个手工 `git clone` PID `24716/25828/26585` 均位于 `/home/jg525/models/*`，迁移时仍在运行；21:48 目录仅 `323/308/329 MiB`，不能视为权重完成 | 等待 clone/LFS 结束；尚未验证固定 revision、权重大小和 SHA256 | 下载结束后运行 `scripts/run_asrur_model_audit.sh` 严格离线验收 |
| 0/2 | SQuTR DATA-13B/13C 六子集内容门禁 | WAITING_USER | `a4c5c02` | 精确探针 old/new lock RC=`73/0`；本机无持有者，用户确认没有任何其他挂载 `/home/jg525` 的运行实例 | DPC 远端遗留锁租约；只阻塞全六子集扩展 | 按 `dpc_lock_support_request.md` 联系平台；不得删除/绕过旧 lock |
| 0/2 | DATA-13C lock owner provenance | COMPLETED | `090abf8` | wrapper 改为非截断打开 lock；成功取得后记录 hostname/PID/PPID/commit/run dir，失败时显示最后记录；4 项专项测试通过 | 该补丁不能解除当前由远端实体持有的旧锁 | 远端锁安全释放并拉取本补丁后再恢复 DATA-13C |
| 0/2 | DATA-13D FiQA/NQ 目标子集门禁 | COMPLETED | `730e0fc` | run=`data13d_squtr_fiqa_nq_validation_20260726_224653`；reuse/validation/FiQA audit/wrapper=`0/0/0/0`；16,400/16,400 音频、manifest/cache SHA256 已固定；FiQA violations 与 split leakage 均为空 | 无；源音频混合 24 kHz/16 kHz，后续 runner 必须显式重采样 | 固化审计 JSON；进入模型验收与 Phase 1/2 runner 门禁 |
| 1 | Nemo base/+Cl 当前缓存实时只读复核 | WAITING_USER | `6760f7f` | attempt 2 run=`asrur_nemo_phase1_audit_20260726_235146` exit=`1/1`；遗留 `TRANSFORMERS_OFFLINE=1` 被 Hub 0.36.0 解释为离线别名，仍在本地内容校验前失败；无模型损坏证据 | 三个 Hub offline alias 的清除补丁已提交，需要远程拉取 | 拉取最新修复后运行 attempt 3 |
| 1 | Nemo inference-only checkpoint 与 model lock | WAITING_USER | `6760f7f` | attempt 2 未创建派生权重或 lock；无 GPU、无模型下载、不覆盖源 checkpoint | 依赖 attempt 3；正式仓库锁只能由本地 Codex 根据返回证据提交 | attempt 3 portable evidence 验证通过后提交 canonical model lock |
| 1 | Nemo 5/25 OEA embedding smoke | TODO | N/A | 计划 1×RTX 4090 24GB | 依赖 model lock 和精确 GPU 命令批准 | 先提交精确 wrapper/config；预计 12–18 GiB |
| 1 | Nemo(+Cl) Clotho 表 2 T2A | TODO | N/A | 需要 1×GPU + CPU metrics | 依赖 smoke gate | 对照 21.57/47.16/60.36 |
| 2 | FiQA corpus/qrels 协议与文档构造锁 | COMPLETED | `730e0fc` | corpus/train/dev/test=`57,638/5,500/500/648`；qrel pairs=`14,166/1,238/1,706`；四条件各 648；ID、规范化文本和 split leakage 检查全通过 | FiQA corpus 有 38 个官方空文档行，按固定 SQuTR loader 保留 | 后续索引不得过滤或合成这 38 行；缓存 manifest 绑定输入 SHA256 |
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
  `lslocks`/`fuser`/`lsof` 均未显示本机持有者。2026-07-26 20:25 的精确探针
  得到 old lock RC=`73`、同目录新 lock RC=`0`，本机仍无 DATA-13C 进程、
  `/proc/locks` 或开放 FD 匹配。因此同目录 `flock` 正常，旧锁由另一台共享
  存储服务器或 DPC 远端租约持有。不得删除或绕过 lock。
- 当前未执行：任何 GPU 计算、TTS、模型推理或真实 gate 训练。D1 FiQA 已完成；
  OEA 模型已完整迁移到 `/home/jg525/models/oea`，D2–D4 正在
  `/home/jg525/models` 下并行 clone，但尚无权重完成证据。
- 下一步：两条 CPU 工作可独立执行：(1) D2–D4 下载完成后运行固定
  revision/LFS/size/SHA256 离线验收；(2) 拉取 `953cb34` 后运行 Nemo base/+Cl
  实时只读复核、inference-only checkpoint 严格准备与 portable model lock。
  DATA-13D 已完成，旧 DATA-13C lock 仅继续影响六子集扩展。portable evidence
  返回并由本地 Codex 固化为 canonical model lock 后，才提交 G1 的精确 GPU 命令。
