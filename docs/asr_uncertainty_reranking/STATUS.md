# ASR-Uncertainty Reranking 状态

## 2026-07-27 ICASSP 范围纠正

- 唯一主实验：冻结 OEA-Nemo3B(+Cl) Top-100 粗召回 + Whisper 4-best
  proxy posterior + 冻结 BGE Cross-Encoder + ASR 不确定性感知候选级动态门控。
- OEA 仅作为论文基线；不再把“完整复现 OEA 全部实验”列为本论文前置任务。
- 必须完成的 OEA 基线与新增实验已逐项固化在
  `docs/asr_uncertainty_reranking/ICASSP_EXPERIMENT_CHECKLIST.md`。
- 当前真实实验完成度：`3/28 = 10.7%`；FiQA 主结果单元为 `0`。已完成的 CPU
  框架和合成测试只计执行准备度，绝不计作研究结果。
- 代码 commit `207bcd0` 已加入本地只读模型适配层；commit `f9fc977` 已加入
  BGE/Whisper/CE、OEA-Nemo 与原始 Omni 的可恢复真实缓存生成器及 Phase-2
  工作量预检。四个声学条件强制独立缓存，并统一使用 FiQA qrels query ID。
  Gold transcript CE 上限允许单假设，但正式 Whisper 路径固定为 4-best。
- 验证：本地全仓 `402` 项 `unittest` 全部通过；未下载、未加载模型、未启动 GPU。
- Phase-2 dry-run attempt 2 已通过：17/17 个步骤、run、wrapper 和调用端退出码
  均为 0，stderr 为空；未加载模型、未使用 GPU、未产生研究指标。
- Phase-2 GPU attempt 1 已结束：`oea_clean=0`、`vanilla_clean=1`、
  `wrapper=1`，无 completion manifest 或 condition metrics。失败是原始 Omni
  BF16 输出在 float32 消费端未通过旧的 `atol=1e-3` 范数断言；不是 OOM。
- commit `92d705f` 的重试前 normalization smoke 未进入模型加载：16 项 CPU
  测试与 RTX 4090 预检通过，但生成器直接入口因仓库根目录未先加入
  `sys.path` 而 exit=`1`，因此 Phase-2 没有启动，归一化修复尚未被真实模型验证。
- commit `ee63772` 的修复后 smoke 已通过：5 条音频、25 条文本、17 项 CPU
  测试均成功。30 行中有 17 行原始范数偏差超过 `1e-3`，最大偏差
  `0.0027647`；float32 cache-boundary L2 后最大偏差为 `1.19e-7`，因此
  attempt 1 根因和最小修复均获真实模型验证。
- Phase-2 attempt 2 已失败并释放 GPU：OEA、vanilla 与 BGE 的 11 组完整缓存可
  复用；`whisper_clean` 的 648/648 条记录均因 FP32 log-mel 输入与 BF16 Whisper
  首层卷积 dtype 不一致而失败。无 completion manifest 或正式指标；不是 OOM，
  也不是坏音频样本。
- commit `5835fab` 已加入 Whisper 浮点输入 dtype 对齐、生成分数 fail-closed、
  连续 8 次失败快速中止，以及完成缓存经 schema/size/SHA256 验证后的跨 commit
  只读复用。全仓 407 项测试和 24 项定向测试通过。
- 单条真实模型门禁已固化为 `scripts/run_asrur_whisper_dtype_smoke.sh`：严格离线，
  只处理首条 FiQA clean 音频并要求恰好 4 个有限 proxy-score 假设；Bash 语法与
  更新后的全仓 408 项测试通过。
- commit `6ef3c2c` 的单条 GPU smoke 已执行并失败：D2–D4 资源严格验收为
  `complete`，原 FP32/BF16 异常未复现，但 Transformers 4.52.4 的 Whisper
  包装层压缩 score rows 后仍保留全局 beam indices，导致 transition-score
  CUDA gather 越界；失败审计已固化，未产生研究指标。
- 最小修复改为显式 English/transcribe/no-timestamps decoder prompt，并只绕过
  Whisper 包装层、调用同一冻结模型的 base `GenerationMixin` 四束搜索；模型、
  checkpoint、数据和候选集合均不变。15 项定向测试和全仓 409 项测试通过。
- commit `ef3a77a` 的第二次单条 smoke 已在 53 秒后失败：base 四束生成完成，
  先前 dtype 与 beam-index 错误均未复现，但至少一个非 special 生成 token 的
  generation transition score 为非有限值，严格校验因此拒绝写出 N-best。
  证据位于 `results/audits/asrur_whisper_transition_score_smoke_failure_20260727.json`。
- 当前修复不删除或截断非有限值，也不伪造后验：保留 beam `sequence_score`，
  对完全相同的四条生成序列做冻结 teacher-forced 前向，以 float32 cross-entropy
  计算实际文本 token 的条件 log-probability。该方法仍明确标记为未校准 proxy；
  缓存身份记录计分方法并禁用 transition-score 复用，失败 JSON 记录精确 stage。
- commit `54d0e72` 的第三次单条 smoke 已通过：同一 `en/fiqa:clean:4641`
  记录生成恰好 4 个假设，beam sequence score 与 teacher-forced proxy score
  全部有限，generation transition score 未使用；strict/wrapper=`0/0`，
  elapsed=`48s`，stderr 与 failure evidence 均为空。成功审计位于
  `results/audits/asrur_whisper_teacher_forced_smoke_success_20260728.json`。
- Phase-2 attempt 3 已在 `whisper_snr_0` 末尾失败：clean/snr_20/snr_10
  均完整，snr_0 保留 `647/648` 个 shard；唯一失败记录
  `en/fiqa:snr_0:10639` 的 4-beam 生成已完成，但至少一个非 special token
  的 teacher-forced 分数非有限。音频为可解码的 3.64 秒、16 kHz 单声道
  PCM，不是缺失文件或整体 GPU 故障；GPU 已释放。失败审计位于
  `results/audits/asrur_phase2_attempt3_single_record_failure_20260728.json`。
- 已实现隔离且不修改缓存的数值诊断：对相同 BF16 beam sequences 比较
  BF16 四假设批量、BF16 单假设和 FP32 upcast 权重批量 teacher-forced
  分数，并逐 token 记录 finite/NaN/±Inf。诊断不会训练、计算研究指标或
  更改已完成 shard。
- 用户已固定后续后台协议为 tmux-only；不再提供 `nohup` 启动命令。当前健康
  运行的 PID `6769` 不为切换工具而中断；新增只读实时监控入口
  `scripts/monitor_asrur_phase2.sh`，显示 run/step elapsed、局部百分比、近期
  吞吐和近似 step ETA。
- Phase-3 无需 dev 选择的 E2/E5/E6/E11 接续 runner 已实现并通过 28 项相关测试；
  只在 Phase-2 四条件汇总为 `GO` 且另行获得 GPU 批准后执行。
- 下一步：在 1×RTX 4090 上运行单记录数值诊断；根据三种计分路径的实测差异
  选择最小修复，再跨 commit 复用旧 11 组缓存、3 个完整 Whisper 条件和
  snr_0 的 647 个完成 shard。

最后更新：2026-07-28（Phase-2 attempt 3 仅一条 0 dB 记录数值失败；等待隔离 GPU 数值诊断）

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
| 0/2 | FiQA 真实模型缓存 producer 与 Phase-2 预检 | COMPLETED | `1ff9fc0` | BGE dense、Whisper 4-best、BGE CE、OEA-Nemo、原始 Omni 与 exact Top-100 均支持断点恢复和不可变 manifest；FiQA dev/test 由 qrels ID 精确隔离；D2–D4 与两套 canonical lock 已完成；全仓 388 项测试通过；dry-run attempt 2 的 17/17 步骤通过 | 无 CPU 侧阻塞；正式结果尚未产生 | 单独申请 Phase-2 GPU |
| 0/2 | 原始 Omni-Embed-Nemotron 模型锁 | COMPLETED | `7cfbfab` | run=`asrur_vanilla_nemo_phase2_audit_20260727_161133`；pipeline/wrapper=`0/0`；6,416-B portable 与 canonical lock SHA256 均为 `eb62da75...cc1c`；CPU-only、未下载、未运行 GPU | 无 | Phase-2 dry-run 使用 `results/model_locks/vanilla_nemotron_3b.json` |
| 0 | B5/B6 主融合的 ASR 路由 | COMPLETED | `b9a81ef` | `[INFERRED][USER-CONFIRMED 2026-07-26]` 主路线固定为 4-best `proxy_posterior`；1-best 只作辅助诊断 | 无 | 正式结果不得根据 test/NQ 表现切换路线 |
| 0 | D1 FiQA 固定资源下载 | COMPLETED | `41efefa` | run=`asrur_d1_fiqa_download_20260726_200414`；download/wrapper exit=`0/0`，manifest=`complete`，5/5 selected files 完成，目标目录约 47 MiB | 无 | DATA-13D 生成目标 manifest 后在同一 run 执行一致性审计 |
| 0 | 模型存储迁移与旧缓存清理 | COMPLETED | `aac088d` | run=`model_cache_migration_v3_20260726_214834`；exit=`0`；OEA 254 files、apparent/allocated bytes 前后完全一致；旧 `/home/jg525/model_cache` 已按用户明确授权删除 | 无 | 后续统一使用 `/home/jg525/models`；历史审计中的旧路径保持原样 |
| 0 | D2–D4 Whisper/BGE 固定资源下载 | COMPLETED | `6e55e2b` | D2 repair run=`asrur_d2_whisper_repair_20260727_134626`；strict/wrapper=`0/0`；Whisper 权重 3,087,130,976 B、SHA256=`a8e94b85...fd95`；D2–D4 selected bytes=`5,823,662,271/5,823,662,271`，errors 为空 | 无 | 资源门禁通过；后续严格离线加载 |
| 0/2 | SQuTR DATA-13B/13C 六子集内容门禁 | WAITING_USER | `a4c5c02` | 精确探针 old/new lock RC=`73/0`；本机无持有者，用户确认没有任何其他挂载 `/home/jg525` 的运行实例 | DPC 远端遗留锁租约；只阻塞全六子集扩展 | 按 `dpc_lock_support_request.md` 联系平台；不得删除/绕过旧 lock |
| 0/2 | DATA-13C lock owner provenance | COMPLETED | `090abf8` | wrapper 改为非截断打开 lock；成功取得后记录 hostname/PID/PPID/commit/run dir，失败时显示最后记录；4 项专项测试通过 | 该补丁不能解除当前由远端实体持有的旧锁 | 远端锁安全释放并拉取本补丁后再恢复 DATA-13C |
| 0/2 | DATA-13D FiQA/NQ 目标子集门禁 | COMPLETED | `730e0fc` | run=`data13d_squtr_fiqa_nq_validation_20260726_224653`；reuse/validation/FiQA audit/wrapper=`0/0/0/0`；16,400/16,400 音频、manifest/cache SHA256 已固定；FiQA violations 与 split leakage 均为空 | 无；源音频混合 24 kHz/16 kHz，后续 runner 必须显式重采样 | 固化审计 JSON；进入模型验收与 Phase 1/2 runner 门禁 |
| 1 | Nemo base/+Cl 当前缓存实时只读复核 | COMPLETED | `cd34cbc` | attempt 3 resource audit exit=`0`：base 22 files/9,423,120,401 B；+Cl 3 files/9,466,834,755 B；缺失/额外/incomplete/symlink/errors 均为空，源 checkpoint SHA256 与 LFS 一致 | 无 | 固定完成证据；不重复下载或更换 checkpoint |
| 1 | Nemo inference-only checkpoint 与 model lock | COMPLETED | `58f80bb` | attempt 4 run=`asrur_nemo_phase1_audit_20260727_004107`：resource/preparation/lock 与 pipeline/wrapper 全部 exit=`0`；59,072,047-B derived SHA256 `2a5bee90...80c4`；canonical lock SHA256 `fd09e4d2...c8c2` | 无 | 锁定资源，进入锁绑定 GPU smoke 准备 |
| 1 | Nemo 5/25 OEA embedding smoke | COMPLETED | `1b23c50` | RTX 4090 run=`oea_nemo3b_clotho_lock_bound_smoke_seed42_20260727_092654`；wrapper/attempt=`0/0`；严格离线；5×512/25×512；544 LoRA；pending=0；峰值 allocated/reserved=9.14/9.49 GiB | 无；两条 Transformers 兼容性警告已原样登记，不修改锁定模型 | 已授权并完成同锁全量 Clotho embedding |
| 1 | Nemo(+Cl) Clotho 全量 embedding | COMPLETED | `4d2341d` | RTX 4090 run=`oea_nemo3b_clotho_embeddings_seed42_20260727_094554`；wrapper/attempt=`0/0`；严格离线；1,045×512 audio、5,225×512 text；pending=0；峰值 allocated/reserved=9.39/10.48 GiB；五个最终工件 SHA256 已固定 | 无；兼容性警告保留，不修改锁定 snapshot | 在 CPU 侧复用固定 embedding，不再加载模型 |
| 1 | Nemo(+Cl) Clotho 表 2 T2A | COMPLETED | `20533b1` | CPU suite=`oea_nemo3b_clotho_t2a_suite_seed42_20260727_125803`；suite/attempt/run=`0/0/0`；stderr 为空；all-caption=`21.7225/47.1196/60.4402`，seed0=`21.6268/46.7943/59.8086` | 严格论文 caption 选择仍 `[MISSING]`，因此两组 `[CODE] close` 分栏且 strict 观察保留 blocked；不得择优 | Phase 1 正确性门禁通过；D2–D4 验收完成后进入 Phase 2 FiQA 一级召回 |
| 1 | Nemo(+Cl) Clotho A2T 方向检查 | COMPLETED | `7cfbfab` | CPU suite=`oea_nemo3b_clotho_a2t_suite_seed42_20260727_161133`；attempt=`0`、stderr 为空；1,045 audio→5,225 captions；R@1/5/10=`26.8900/51.3876/65.2632` | 无；这是 `[CODE]` 方向扩展，不是论文表格结果 | OEA-B2 完成；进入 FiQA frozen-index 方向 |
| 2 | FiQA corpus/qrels 协议与文档构造锁 | COMPLETED | `730e0fc` | corpus/train/dev/test=`57,638/5,500/500/648`；qrel pairs=`14,166/1,238/1,706`；四条件各 648；ID、规范化文本和 split leakage 检查全通过 | FiQA corpus 有 38 个官方空文档行，按固定 SQuTR loader 保留 | 后续索引不得过滤或合成这 38 行；缓存 manifest 绑定输入 SHA256 |
| 2 | B1/B2/B3 四条件一级召回 | FAILED | `c328dfd` | attempt 1 run=`asrur_phase2_frozen_retrieval_execute_20260727_175132`；wrapper=`1`；OEA clean=`0`，vanilla clean=`1`；无 completion manifest、无 condition metrics；失败证据已固化 | vanilla BF16 adapter 输出未通过旧的 `atol=1e-3` float32 范数断言；不是 OOM | 使用记录 pre/post 范数的 float32 cache-boundary L2 最小补丁，在新 commit/cache root 重试 |
| 2 | Phase-2 CPU dry-run attempt 1 | FAILED | `7a098dd` | run=`asrur_phase2_frozen_retrieval_dry-run_20260727_164746`；resolve OEA=`0`，resolve vanilla=`1`，wrapper=`1`；`ModuleNotFoundError: scripts`；未加载模型/GPU | 直接脚本入口未把仓库根目录加入 `sys.path` | 拉取最小补丁后使用新 commit/cache root 重试；失败证据不删除 |
| 2 | Phase-2 CPU dry-run attempt 2 | COMPLETED | `1ff9fc0` | run=`asrur_phase2_frozen_retrieval_dry-run_20260727_172606`；17/17 step exit=`0`，run/wrapper/caller=`0/0/0`，stderr 为空；CPU-only，未加载模型/GPU，未产生研究指标 | 无 | 请求 1×RTX 4090 24GB 正式执行 |
| 2 | Phase-2 normalization repair smoke attempt 1 | FAILED | `92d705f` | host=`bitahub-a20615852222705664348817`；16 项 CPU 测试与 RTX 4090/BF16 预检通过；smoke/retry=`1/14`；`ModuleNotFoundError: scripts`；未加载模型，Phase-2 未启动 | 生成器作为绝对路径直接执行时，仓库根目录未在 repository-local imports 前加入 `sys.path` | 拉取最小入口补丁；先重跑 smoke，成功后才启动 Phase-2 |
| 2 | Phase-2 normalization repair smoke attempt 2 | COMPLETED | `ee63772` | 17 项 CPU 测试通过；5×2048 audio、25×2048 text；smoke=`0`；pre 最大范数偏差 `0.0027647`，30 行中 17 行超过旧阈值；post 最大偏差 `1.19e-7`；峰值 allocated/reserved=`9,757,201,408/10,139,729,920` B | 无；三条非阻塞兼容性/资源警告原样保留 | 以同一 commit 启动 Phase-2 |
| 2 | B1/B2/B3 四条件一级召回 attempt 2 | FAILED | `ee63772` | run=`asrur_phase2_frozen_retrieval_execute_20260727_193044`；OEA/vanilla 四条件与 BGE corpus/dev 的 11 组 cache manifest 完整；`whisper_clean` 648/648 失败；wrapper=`1`；GPU 已释放；无正式指标 | FP32 `input_features` 与 BF16 Whisper conv bias dtype 不一致；不是 OOM 或坏样本 | 修复 commit `5835fab`；保留旧证据并通过校验后只读复用完整缓存 |
| 2 | Phase-2 Whisper dtype repair smoke attempt 1 | FAILED | `6ef3c2c` | host=`bitahub-a20618022373814272490966`；D2–D4 strict audit complete；单条 `en/fiqa:clean:4641`；原 dtype 异常未复现；smoke=`1`；无正式指标 | Transformers 4.52.4 Whisper wrapper 压缩 score rows 后仍使用全局 beam indices，CUDA transition-score gather 越界 | 失败审计已固化；保留同模型/协议，改用 base `GenerationMixin` 四束搜索 |
| 2 | Phase-2 Whisper base-generation repair smoke attempt 2 | FAILED | `ef3a77a` | run=`asrur_whisper_dtype_smoke_20260727_234354`；同一记录；elapsed=`53s`；base 四束生成完成；dtype 与 beam gather 错误未复现；wrapper=`1`；无正式指标 | 有效非 special token 的 generation transition score 含非有限值；不允许过滤后冒充概率 | 失败证据已固化；移除对 generation transition score 的依赖 |
| 2 | Phase-2 Whisper teacher-forced proxy repair smoke | COMPLETED | `54d0e72` | run=`asrur_whisper_dtype_smoke_20260728_001309`；同一记录恰好 4 个假设；beam sequence/proxy score 均有限；transition score 未使用；wrapper=`0`；elapsed=`48s`；stderr/failure 为空 | 无；这是正确性门禁，不是研究指标 | 用新的 cache/result root 启动 attempt 3；旧失败与 11 组完整缓存不覆盖 |
| 2 | B1/B2/B3 四条件一级召回 attempt 3 | FAILED | `bc4450a` | clean/snr_20/snr_10 Whisper 完整；snr_0=`647/648` shards；唯一失败=`en/fiqa:snr_0:10639`；wrapper=`1`；无 completion manifest/正式指标；GPU 已释放 | N-best artifact validation 检出至少一个非 special token 的 teacher-forced 分数非有限；具体数值路径尚未确定 | 保留全部 cache；禁止过滤/clamp/fallback；运行隔离数值诊断 |
| 2 | Whisper 单记录 BF16/FP32 数值诊断 | COMPLETED | `c294730` | run=`asrur_whisper_numeric_diagnostic_20260728_012926`；wrapper=`0`；elapsed=`33s`；同一 `en/fiqa:snr_0:10639` 的 beam sequence score、BF16 batched、BF16 per-hypothesis、FP32-upcast 三路有效 token 分数全部有限；峰值 allocated=`6,448,808,960` B | 原失败不是音频或某条固定评分路径的稳定错误；证据符合瞬态同协议低精度数值异常，但不证明底层 kernel 根因 | 使用严格分片导入和最多 3 次的同协议数值重试；禁止过滤、clamp、替换分数或回退协议 |
| 2 | Phase-2 attempt 4 严格恢复与完成 | COMPLETED | `d9baf22` | run=`asrur_phase2_frozen_retrieval_execute_20260728_093950`；wrapper=`0`；44/44 steps=`0`；14 个完整 cache 复用；647 个 snr_0 shards 逐条导入；缺失记录首次成功；最终 648 shards、无新 failure、completion=`complete` | 无执行错误；真实指标触发协议门禁 | 固化失败结果并停止 Phase 3 |
| 2 | Phase-2 四条件结果与 Go/No-Go 审计 | COMPLETED | `d9baf22` | audit SHA256=`f919a84f...eb8e`；四条件均含 648 queries 与 B1/B2/B3/U1；原始 Omni clean nDCG@10=`0.258495`，Gold+BGE=`0.405853` | OEA clean Recall@100=`0.002561`，Whisper+BGE clean nDCG@10=`0`；每个条件的三项门禁全部失败 | 运行 CPU-only transcript/WER 与 embedding-geometry 诊断 |
| 2 | Go/No-Go 审查 | FAILED | `d9baf22` | overall=`NO_GO_REQUIRES_USER_DECISION`；未授权改变候选生成；OEA Top-100 oracle nDCG@10 仅 `0.001972–0.003761` | 尚未确定 OEA 是 checkpoint/protocol 错误还是真实跨域失败；Whisper cache 内容也需审计 | 在证据明确前不运行 CE、融合或门控训练；任何换 checkpoint/ASR 协议/双路召回都先问用户 |
| 3 | E2/E5/E6/E11：1-best CE、4-best equal/max、Gold CE | BLOCKED | 本提交 | 四条件共享冻结 4×100 CE 矩阵、单独 Gold 单假设 CE、严格无 test 调参评测和断点恢复 runner 已实现；28 项相关测试、Python compile 与 Bash 语法通过 | 依赖 Phase-2 汇总 GO 和新的 GPU 批准 | 复用唯一 OEA Top-100，不重新召回；Gold 不伪造 ASR 后验 |
| 3 | E3/E4/E7：固定融合、RRF、proxy posterior | BLOCKED | `8f28c55` | CPU 聚合与 dev 选择已有测试实现；尚无真实 CE 缓存或 dev 选择结果 | 必须使用 FiQA dev 选择 | 不得用 FiQA test qrels 选择超参数 |
| 4 | 4-best、proxy posterior、不确定性实验 | BLOCKED | `8f28c55` | 缓存装配、聚合、特征、dev 选择、正式评测实现完成；模型推理未开始 | 依赖 Whisper 下载、Phase 3 缓存和 GPU 批准 | 补模型 adapter 后生成正式缓存 |
| 4 | FiQA train/dev TTS/speaker/noise 协议 | BLOCKED | N/A | 未下载/合成 | TTS、许可证、speaker/noise 隔离未批准 | 单独提交具体计划和预算 |
| 4 | Query/candidate gate 训练 | BLOCKED | `8f28c55` | 三 seed、小 MLP、multi-positive listwise、zero-positive 审计已实现；仅合成测试 | 依赖 TTS 协议和用户真实训练批准 | 只训练小 gate；上游模型保持冻结 |
| 5 | FiQA 三 seed/消融/bootstrap/分析 | BLOCKED | `8f28c55` | A1–A9、paired bootstrap、噪声/gate/统一 CSV 已实现；无真实数值 | 依赖 Phase 4 正式缓存 | 完整四条件正式评测 |
| 6 | NQ zero-shot | BLOCKED | N/A | 未开始 | 依赖 FiQA 六项门禁和新 GPU 批准 | 不在 NQ 调参 |
| 7 | 效率、表格、图和最终研究报告 | TODO | N/A | 未开始 | 依赖主实验 | 区分事实、推断、失败与局限 |

## 当前焦点

- 当前阶段：Phase 0、数据/模型门禁、OEA Phase 1 和 Phase-2 CPU dry-run
  已完成。真实 FiQA 模型推理、TTS 与 gate 训练仍未开始。
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
- 当前未执行：正式 Whisper/BGE/OEA/Omni FiQA 推理、TTS 或真实 gate 训练。
  D1–D4、两套模型锁及 Phase-2 CPU dry-run 均已完成。
- 下一步：单独申请 1×RTX 4090 24GB，运行 Phase 2 FiQA 四条件冻结一级召回；
  未取得真实结果前不改变候选生成方式，也不进入 TTS/gate 训练。
