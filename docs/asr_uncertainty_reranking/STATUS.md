# ASR-Uncertainty Reranking 状态

## 2026-07-27 ICASSP 范围纠正

- 唯一主实验：冻结 OEA-Nemo3B(+Cl) Top-100 粗召回 + Whisper 4-best
  proxy posterior + 冻结 BGE Cross-Encoder + ASR 不确定性感知候选级动态门控。
- OEA 仅作为论文基线；不再把“完整复现 OEA 全部实验”列为本论文前置任务。
- 必须完成的 OEA 基线与新增实验已逐项固化在
  `docs/asr_uncertainty_reranking/ICASSP_EXPERIMENT_CHECKLIST.md`。
- 当前真实实验完成度：`2/28 = 7.1%`；FiQA 主结果单元为 `0`。已完成的 CPU
  框架和合成测试只计执行准备度，绝不计作研究结果。
- 代码 commit `207bcd0` 已加入本地只读模型适配层；commit `dd44bc4` 已加入
  BGE/Whisper/CE、OEA-Nemo 与原始 Omni 的可恢复真实缓存生成器及 Phase-2
  工作量预检。四个声学条件强制独立缓存，并统一使用 FiQA qrels query ID。
  Gold transcript CE 上限允许单假设，但正式 Whisper 路径固定为 4-best。
- 验证：本地全仓 `380` 项 `unittest` 全部通过；未下载、未加载模型、未启动 GPU。
- 当前阻塞：D2 Whisper `model.safetensors` 远程断点恢复与严格 SHA256 验收；
  D3/D4 已完成。D2 完成后先执行独立 GPU 批准的 Phase-2 召回，不等待 TTS。
- 下一步：D2 严格验收完成后执行 canonical lock 与 Phase-2 dry-run，然后向
  用户提交一次明确的 GPU 任务申请。

最后更新：2026-07-27（D2 单文件断点恢复正在远程 CPU 运行）

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
| 0/2 | FiQA 真实模型缓存 producer 与 Phase-2 预检 | COMPLETED | `dd44bc4` | BGE dense、Whisper 4-best、BGE CE、OEA-Nemo、原始 Omni 与 exact Top-100 均支持断点恢复和不可变 manifest；全仓 380 项测试通过；未运行 GPU | D2 尚在下载；原始 Omni canonical lock 尚待 CPU 生成 | D2 验收后做 dry-run；随后单独申请 GPU |
| 0 | B5/B6 主融合的 ASR 路由 | COMPLETED | `b9a81ef` | `[INFERRED][USER-CONFIRMED 2026-07-26]` 主路线固定为 4-best `proxy_posterior`；1-best 只作辅助诊断 | 无 | 正式结果不得根据 test/NQ 表现切换路线 |
| 0 | D1 FiQA 固定资源下载 | COMPLETED | `41efefa` | run=`asrur_d1_fiqa_download_20260726_200414`；download/wrapper exit=`0/0`，manifest=`complete`，5/5 selected files 完成，目标目录约 47 MiB | 无 | DATA-13D 生成目标 manifest 后在同一 run 执行一致性审计 |
| 0 | 模型存储迁移与旧缓存清理 | COMPLETED | `aac088d` | run=`model_cache_migration_v3_20260726_214834`；exit=`0`；OEA 254 files、apparent/allocated bytes 前后完全一致；旧 `/home/jg525/model_cache` 已按用户明确授权删除 | 无 | 后续统一使用 `/home/jg525/models`；历史审计中的旧路径保持原样 |
| 0 | D2–D4 Whisper/BGE 固定资源下载 | RUNNING_REMOTE | `6e55e2b` | D3/D4 已严格通过；D2 repair run=`asrur_d2_whisper_repair_20260727_134626`，PID `4940`，CPU-only，启动成功且使用 `.part` 断点续传；13:48 约 92.9 MiB/2.944 GiB，strict audit 尚未开始 | 仅等待 D2 的 3,087,130,976-byte `model.safetensors` 下载、SHA256 校验与自动全量验收 | 不重复启动；任务退出后收集 repair/audit 两级退出码与最终 SHA256 |
| 0/2 | SQuTR DATA-13B/13C 六子集内容门禁 | WAITING_USER | `a4c5c02` | 精确探针 old/new lock RC=`73/0`；本机无持有者，用户确认没有任何其他挂载 `/home/jg525` 的运行实例 | DPC 远端遗留锁租约；只阻塞全六子集扩展 | 按 `dpc_lock_support_request.md` 联系平台；不得删除/绕过旧 lock |
| 0/2 | DATA-13C lock owner provenance | COMPLETED | `090abf8` | wrapper 改为非截断打开 lock；成功取得后记录 hostname/PID/PPID/commit/run dir，失败时显示最后记录；4 项专项测试通过 | 该补丁不能解除当前由远端实体持有的旧锁 | 远端锁安全释放并拉取本补丁后再恢复 DATA-13C |
| 0/2 | DATA-13D FiQA/NQ 目标子集门禁 | COMPLETED | `730e0fc` | run=`data13d_squtr_fiqa_nq_validation_20260726_224653`；reuse/validation/FiQA audit/wrapper=`0/0/0/0`；16,400/16,400 音频、manifest/cache SHA256 已固定；FiQA violations 与 split leakage 均为空 | 无；源音频混合 24 kHz/16 kHz，后续 runner 必须显式重采样 | 固化审计 JSON；进入模型验收与 Phase 1/2 runner 门禁 |
| 1 | Nemo base/+Cl 当前缓存实时只读复核 | COMPLETED | `cd34cbc` | attempt 3 resource audit exit=`0`：base 22 files/9,423,120,401 B；+Cl 3 files/9,466,834,755 B；缺失/额外/incomplete/symlink/errors 均为空，源 checkpoint SHA256 与 LFS 一致 | 无 | 固定完成证据；不重复下载或更换 checkpoint |
| 1 | Nemo inference-only checkpoint 与 model lock | COMPLETED | `58f80bb` | attempt 4 run=`asrur_nemo_phase1_audit_20260727_004107`：resource/preparation/lock 与 pipeline/wrapper 全部 exit=`0`；59,072,047-B derived SHA256 `2a5bee90...80c4`；canonical lock SHA256 `fd09e4d2...c8c2` | 无 | 锁定资源，进入锁绑定 GPU smoke 准备 |
| 1 | Nemo 5/25 OEA embedding smoke | COMPLETED | `1b23c50` | RTX 4090 run=`oea_nemo3b_clotho_lock_bound_smoke_seed42_20260727_092654`；wrapper/attempt=`0/0`；严格离线；5×512/25×512；544 LoRA；pending=0；峰值 allocated/reserved=9.14/9.49 GiB | 无；两条 Transformers 兼容性警告已原样登记，不修改锁定模型 | 已授权并完成同锁全量 Clotho embedding |
| 1 | Nemo(+Cl) Clotho 全量 embedding | COMPLETED | `4d2341d` | RTX 4090 run=`oea_nemo3b_clotho_embeddings_seed42_20260727_094554`；wrapper/attempt=`0/0`；严格离线；1,045×512 audio、5,225×512 text；pending=0；峰值 allocated/reserved=9.39/10.48 GiB；五个最终工件 SHA256 已固定 | 无；兼容性警告保留，不修改锁定 snapshot | 在 CPU 侧复用固定 embedding，不再加载模型 |
| 1 | Nemo(+Cl) Clotho 表 2 T2A | COMPLETED | `20533b1` | CPU suite=`oea_nemo3b_clotho_t2a_suite_seed42_20260727_125803`；suite/attempt/run=`0/0/0`；stderr 为空；all-caption=`21.7225/47.1196/60.4402`，seed0=`21.6268/46.7943/59.8086` | 严格论文 caption 选择仍 `[MISSING]`，因此两组 `[CODE] close` 分栏且 strict 观察保留 blocked；不得择优 | Phase 1 正确性门禁通过；D2–D4 验收完成后进入 Phase 2 FiQA 一级召回 |
| 2 | FiQA corpus/qrels 协议与文档构造锁 | COMPLETED | `730e0fc` | corpus/train/dev/test=`57,638/5,500/500/648`；qrel pairs=`14,166/1,238/1,706`；四条件各 648；ID、规范化文本和 split leakage 检查全通过 | FiQA corpus 有 38 个官方空文档行，按固定 SQuTR loader 保留 | 后续索引不得过滤或合成这 38 行；缓存 manifest 绑定输入 SHA256 |
| 2 | B1/B2/B3 四条件一级召回 | BLOCKED | `dd44bc4` | 真实 runner 已实现并通过测试；每条件独立 648 query，统一 qrels ID | D2 严格验收、原始 Omni lock 与独立 GPU 批准 | 报告 nDCG/MRR/Recall/Oracle |
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
- 当前未执行：TTS、Whisper/BGE 推理或真实 gate 训练。D1 FiQA、OEA Phase 1、
  D3 和 D4 已完成；D2 的单文件断点恢复正在远程 CPU 后台运行。
- 下一步：等待 D2 repair 自动完成大小/SHA256 校验和 D2–D4 严格离线验收。
  验收全通过后，先建立 D2–D4 canonical model locks 和 Phase 2 dry-run；
  再单独申请 FiQA 一级召回所需 GPU，不复用 Phase 1 的 GPU 授权。
