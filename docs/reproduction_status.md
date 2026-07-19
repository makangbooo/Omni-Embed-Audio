# Omni-Embed-Audio 复现状态

最后更新：2026-07-19（Asia/Shanghai）

状态值仅使用：`TODO`、`IN_PROGRESS`、`WAITING_USER`、`RUNNING_REMOTE`、`COMPLETED`、`FAILED`、`BLOCKED`。

| 阶段 | 任务 | 状态 | 当前 Commit | 远程状态 | 阻塞原因 | 下一步 |
|---|---|---|---|---|---|---|
| 0 项目审计 | 完整阅读论文正文/附录/表/图/脚注 | COMPLETED | `48d7a50` | fork 已同步 | 无 | 固定 inventory |
| 0 项目审计 | 审计 README、依赖、训练/评测、数据、UIQ、HN、指标 | COMPLETED | `48d7a50` | fork 已同步 | 无 | 用户确认审计结论 |
| 0 项目审计 | 31 条实验/图/派生分析 inventory | COMPLETED | `48d7a50` | fork 已同步 | 无 | 用户确认复现边界 |
| 0 GitHub | 建立 `repro/oea-full` 分支 | COMPLETED | `48d7a50` | 本地与 fork 分支均存在 | 无 | 后续提交仅推送该分支 |
| 0 GitHub | commit 并 push 第一阶段审计 | COMPLETED | `48d7a502e279bd9a2a3195352163626fb1781a3b` | `origin/repro/oea-full` 已建立 | 无 | 保持本地与远程实验 commit 一致 |
| 0 编排 | 统一复现入口与 CPU/GPU 阶段门禁 | COMPLETED | `d13b115` | 公共入口、精确阶段注册表、默认只规划/显式执行、长任务确认和干净 worktree 门禁已实现；baseline 计划已展开为 vanilla CPU 模型锁 → A100 smoke → A100 full → CPU 四协议 metrics 与独立 CLAP blocker；全量 227 项测试通过；尚未在远程执行这些新阶段 | 无；入口完成不代表任何 blocked 实验已解锁或产生复现值 | 仅在固定资源与证据满足后更新叶子阶段状态；当前仍只等待 DATA-04 |
| 1 环境 | CPU-01：CPU/内存/存储/编译器/网络审计 | COMPLETED | `20770d0` | 远程完成；原始日志未进 Git | Zenodo 返回 503；容器 `nproc=2` | 在共享目录挂载到 A100 后执行 GPU-01 |
| 1 环境 | GPU-01：A100/Driver/CUDA/NCCL 基础审计 | COMPLETED | `905e84d` | 1×A100-SXM4-80GB 远程完成；原始日志未进 Git | 多卡 GPU–GPU 拓扑留待 DDP smoke | 固定环境候选并执行 GPU-02 |
| 1 环境 | 生成环境候选、安装脚本与无模型检查 | COMPLETED | `c61fecb` | fork 已同步 | 完整 transitive lock 必须在 Linux solve 后生成 | 固定首批模型资源 |
| 1 环境 | CPU-02：创建 `oea-repro` 并生成 resolved lock | COMPLETED | `11f20d2` | `pip check`、必需接口、仓库导入、音频 I/O/重采样均通过；resolved artifacts 已生成 | 无 | 在单卡实例执行 GPU-02 |
| 1 环境 | GPU-02：CUDA/BF16/NCCL 验证 | COMPLETED | `711ef28` | 1×A100-SXM4-80GB 验证通过；原始日志保存在共享目录 | 无；`flash-attn` 为非必需可选项 | 下载首批固定 revision 模型资源 |
| 1 环境 | CPU-03：安装 DATA-02 的 7-Zip 工具 | COMPLETED | `a54cf5e` | `7zip=26.02` 已在共享 `oea-repro` 可用；install/wrapper exit 均为 0 | 无；Conda 报告目标包已安装，未发生依赖 transaction | 启动并行资源下载 |
| 2 官方权重 | MODEL-01：Qwen2.5-Omni-3B + OEA-Qwen3B-Cl | COMPLETED | `bba6084` | 断点续传完成；两个 asset 和全部 LFS SHA256 通过；无临时文件/下载进程 | 无；首次失败证据保留在旧运行目录 | 审计 checkpoint 内部结构 |
| 2 官方权重 | MODEL-02：OEA-Qwen3B-AC checkpoint | COMPLETED | `18e4c3b` | 固定 revision 的 3 个文件、9,466,844,378 bytes 全部通过逐文件内容审计；`step_350.pt` 提取为 59,069,203-byte inference-only 权重，SHA256 `b1d0f559711b70f5a80dbdeb8cd46d80ed7b38b8e5524b9a871878d8bcd5f101` | 无；原始 checkpoint 与派生权重均未提交到 Git | GPU 服务器拉取已提交的 5,262-byte 模型锁后运行独立锁绑定 smoke |
| 2 官方权重 | MODEL-03：Nemotron-3B base + OEA-Nemo3B AC/Cl | IN_PROGRESS | `48c9b3c` | 已知 base/AC 完整，Cl 曾保留 partial；最近检查的主容器无活动下载进程 | CAS 链路不稳定；最终 Cl 文件尚无完成证据 | 后续单独核对 manifest、partial 与 LFS SHA256，再断点续传 |
| 2 官方权重 | MODEL-04：Qwen2.5-Omni-7B base + OEA-Qwen7B AC/Cl | IN_PROGRESS | `48c9b3c` | 已知 base 完整、AC 曾保留 partial、Cl 未有开始证据；最近检查的主容器无活动下载进程 | CAS 链路不稳定；跨实例状态未确认 | 后续单独核对全部目录与 manifest，再断点续传 |
| 2 官方权重 | MODEL-03/04 只读完整性审计器 | COMPLETED | `f4309a7` | 远程尚未运行；本地实现及 105 项完整测试通过 | 无；工具完成不代表六个 asset 已完成 | DATA-04 保持为唯一用户步骤；随后在 CPU 侧运行审计并按 asset 决定续传 |
| 2 官方权重 | 六变体按变体资源审计入口 | COMPLETED | `d2f066c` | 注册表驱动地只选择一个 base 与一个 checkpoint；Qwen3B AC 跨 MODEL-01/02 manifest 已覆盖；报告固定 scope/registry/逐文件 Git-LFS 身份，完整 172 项测试通过 | 尚未在远程对六个变体逐一运行；工具完成不代表资源完整 | DATA-04 后按已知下载状态选择变体，在 CPU 侧运行只读审计；超过 30 分钟的读取任务执行前单独汇报 |
| 2 官方权重 | 六变体原始 checkpoint 固定、结构审计与推理权重提取器 | COMPLETED | `ee28245` | 六个不可变 revision 的真实文件名/字节数/LFS SHA256 已固定；FakeTensor LoRA 子集统计、逐 checkpoint 非覆盖提取、逐张量等值复核和失败工件已实现；完整 160 项测试通过 | 除已验证的 Qwen3B-Cl 外，其余五个派生权重尚未在远程生成；工具完成不代表原始下载已完整 | 保持 DATA-04 为唯一用户步骤；之后对已完成的原始 checkpoint 先做 CPU 结构审计，再凭派生 SHA256 生成模型专属评测配置 |
| 2 官方权重 | 六变体可移植模型资源锁生成器 | COMPLETED | `18e4c3b` | Qwen3B-Cl 与 Qwen3B 两个变体均已完成真实远程审计、派生权重提取和正式模型锁；基础模型全文件、原始/派生 checkpoint SHA256 及实测 LoRA 结构均已锁定 | Nemo3B AC/Cl 与 Qwen7B AC/Cl 四个变体仍需完成资源审计和模型锁 | 先完成 Qwen3B 锁绑定 GPU smoke，再按已下载资源状态选择其余变体 |
| 2 官方权重 | 官方模型 CPU 三阶段流水线与已有派生权重只读复核 | COMPLETED | `18e4c3b` | Qwen3B-Cl 的 5,292-byte 锁 SHA256 为 `790d12cc...e36`；Qwen3B 运行 `logs/official_model_pipeline_oea_qwen3b_20260719_215427`，资源审计、checkpoint 准备、模型锁三阶段 exit 均为 0，5,262-byte 锁 SHA256 `b6c819311206f583e861991c0678b0b3b61cc2fd346c98c1c30f67c62d0127fa` | 无；两套派生权重均只保留在远程模型缓存 | GPU 服务器拉取 Qwen3B 模型锁后在 RTX 4090 上运行同锁小批 embedding smoke |
| 2 官方权重 | 正式评测协议与可移植模型锁强绑定 | COMPLETED | `97dab6a` | Caption/UIQ GPU 生成器、CPU 套件及 5/25 smoke 均强制解析已提交模型锁；三个正式 Qwen3B wrapper 使用不可绕过的单 CUDA GPU + BF16 能力门禁并记录型号/显存到 `gpu_preflight.json`；4090 与 A100 均被单元覆盖，完整 233 项测试通过 | 无；Qwen3B 正确性评测不再把 A100 误写成必要条件 | 保持锁和协议哈希不变，后续变体先做独立 smoke |
| 2 官方权重 | 5 个 Clotho 样例 Qwen3B-Cl smoke | COMPLETED | `fba8c3c` | 运行 `..._20260716_130619` 严格离线成功；5 音频/5 查询、544 LoRA、双 head、512 维归一化 embedding 全部通过；峰值 9.141 GiB allocated | 无；首次失败 `..._20260716_125636` 仍保留 | 固定小型结果摘要并进入检索指标单元测试 |
| 2 官方权重 | Qwen3B-Cl 正式生成器锁绑定 5/25 GPU smoke | COMPLETED | `97dab6a` | RTX 4090 运行 `...lock_bound_smoke_seed42_20260719_174309`：能力门禁 complete、5×512/25×512、allocated 9.14 GiB、reserved 9.45 GiB、exit 0；模型锁 SHA256 为 `790d12cc...e36` | 无；旧外层 A100 断言与继续执行的矛盾证据仍保留 | 已据此授权并完成同锁全量生成 |
| 2 官方权重 | Qwen3B 正式生成器锁绑定 5/25 GPU smoke | COMPLETED | `2de1ee7` | RTX 4090 运行 `...lock_bound_smoke_seed42_20260719_224120`：能力门禁 complete、5×512/25×512、allocated 9.14 GiB、reserved 9.45 GiB、pending=0、exit 0；模型锁 SHA256 为 `b6c81931...127fa` | 无；两条 Transformers 兼容性提示为非致命警告 | 固化小型审计记录，并用该证据约束同变体 Clotho 全量 embedding |
| 2 官方权重 | Qwen3B 全量 checkpoint embedding 生成器 | WAITING_USER | `f7ac255` | 首次尝试 `..._20260719_231724/attempt_20260719_231724` 在门禁模块导入阶段 exit 1，模型和 embedding 均未加载；原始错误已固化。最小 repository-root bootstrap 修复和真实入口回归测试已提交，完整 243 项测试通过 | 等待远程 RTX 4090 拉取修复 commit 后在同一 RUN_ID 下创建新 attempt | 复用已完成 smoke 的 `generation_metrics.json`，重新启动约 15–30 分钟全量生成；保留首次失败 attempt，不覆盖任何证据 |
| 2 官方权重 | Tables 2/3：单个 3B 官方权重 T2A/T2T | IN_PROGRESS | `97dab6a` | Qwen3B-Cl/Clotho 闭环已完成：T2A `[CODE]` all-caption R@1/5/10=`22.7368/49.4737/63.3110`；T2T `[CODE]` seed0=`64.4019/76.1722/80.7656`；另保留两套预声明协议；12 个观测均分栏写入结果汇总 | `[MISSING]` 论文 caption 选择、T2T self/tie 口径；AudioCaps/MECAT 音频候选集和其余模型仍未准备 | 先复现同一模型的 Clotho 正向 UIQ，再扩展其他权重和数据集 |
| 2 官方权重 | Qwen3B-Cl 全量 checkpoint embedding 生成器 | COMPLETED | `97dab6a` | RTX 4090 真实运行 `...embeddings_seed42_20260719_181621`：1,045 audio/5,225 caption 全部完成、512 维、pending=0、exit 0、63 MB；峰值 allocated 9.307 GiB/reserved 10.115 GiB；四个最终工件 SHA256 已固定 | 论文 `passage:` 与公开代码 no-prefix 冲突已显式标记；公开代码协议不能冒充严格论文协议 | 大型 embedding 保留远程路径与哈希，供 UIQ CPU 套件复用 |
| 2 官方权重 | Qwen3B-Cl/Clotho T2A/T2T 四协议 CPU 评测套件 | COMPLETED | `97dab6a` | CPU 运行 `...retrieval_suite_seed42_20260719_185937`：suite 与四协议 exit 均为 0，525 MB；T2A 两口径、T2T `[CODE]` 与 `[INFERRED]` 敏感性均已分栏；小型摘要 SHA256 `37fb82db...a4a0`，结果 CSV SHA256 `b16a86e5...836` | 论文未公开 caption 选择与 T2T self/tie 口径，四协议不得择优冒充论文口径；严格论文观察继续标为 blocked | 已将 12 个 close 观察及 6 个 strict blocked 观察统一写入结果汇总 |
| 2 指标 | Canonical T2A/T2T/UIQ embedding evaluator | COMPLETED | `43158ae` | 确定性 ID 检索、caption 多正例/排除 self、显式 query 子集、完整排名和严格输入校验已实现；相关 15 项测试通过 | 论文未公开 T2T caption 选择与 tie 口径；已在协议文档标为 `[MISSING]` | 接入 checkpoint embedding 生成器 |
| 2 指标 | 可审计 embedding 实验输出 runner | COMPLETED | `5ed57a7` | CPU-only runner 已保存固定 embedding、metadata、完整排名、输入/工件 SHA256、seed、Git/环境/命令和失败证据；相关测试总计 27 项通过 | 无；正式运行要求干净 worktree 和显式协议来源 | 用 Qwen3B-Cl + Clotho evaluation 完成首个正式 T2A/T2T 闭环 |
| 2 结果管理 | 论文值/复现观测/证据统一汇总契约 | COMPLETED | `35e064f` | Tables 1–17 与 910 个论文指标仍保持 910/910 转录匹配；当前汇总含 30 个观察：24 个 Qwen3B-Cl/Clotho 分栏 close 与 6 个 strict blocked；本地证据、远程工件路径/哈希、delta 和状态计数均已生成 | 无；close 为显式人工审阅结论，未由生成器自动阈值判定 | 后续正式实验继续仅凭小型固定 JSON 证据追加观察 |
| 2 指标 | Figure 3 / Table 17 指标单元测试 | COMPLETED | `88a7d1f` | R@k、Δ-Rank、HNSR、HNSR@k、TFR、TFR-HN@k 已按论文公式实现；7 个合成测试及完整 24 项测试通过 | 无；正式表 17 仍缺 target-HN audio ID | 保留为 canonical negative evaluator 的公式回归测试 |
| 2 指标 | 显式 target/HN 的可审计 negative embedding evaluator | COMPLETED | `cd4d00b` | 严格 query/target/HN ID 覆盖、optimistic rank、确定性完整排序、per-query 证据、输入/输出 SHA256、失败保留和 CPU wrapper 已实现；完整 116 项测试通过 | 工具完成不等于表 17 已解锁；发布数据仍无 HN audio ID | 获得作者 pairing 后才运行正式表 17；否则仅运行单独标记的重建协议 |
| 2 UIQ | Tables 12–15 正向 UIQ | IN_PROGRESS | `35e064f` | Qwen3B-Cl/Clotho 首个完整闭环完成：RTX 4090 生成四类 4,180×512 embedding；CPU suite `...positive_uiq_suite_seed42_20260719_211752` 四协议与 wrapper 共五个 exit 均为 0、100 MB；12 个 R@k 与论文绝对差均不超过 0.575 个百分点，全部作为 `[CODE] close` 写入统一结果表；MRR/DCG 作为额外审计指标保留 | 当前变体/数据集无阻塞；完整 Tables 12–15 仍缺其余五个 OEA 变体、四个 CLAP、AudioCaps/MECAT，且 MECAT 847/848 口径未决 | 固化 12 个观察和 suite 哈希；随后在 CPU 侧重新审计已下载模型/数据，选择下一个无需训练的官方权重评测闭环 |
| 2 Negative | Tables 4/17 否定查询 | BLOCKED | N/A | 未开始 | 发布文件无 HN audio ID | 请求作者 pairing 或审计式重建 |
| 3 数据 | WavCaps duration + leakage blocklists | IN_PROGRESS | `3c5b1e9` | 本地两次全量 metadata pass 已完成；精确复现 173 个 AudioCaps 与 638 个 Clotho 文件名重叠 | 论文 `<=31s` 与公开元数据计数冲突；精确 filtered manifest、Clotho 消歧和论文 blocklist 未发布 | 远程执行 DATA-08/09 复算；训练前再准备音频 |
| 3 数据 | DATA-08：固定 WavCaps metadata 下载 | TODO | `3c5b1e9` | 8 个固定文件、176,863,095 bytes、逐文件 SHA256/LFS ID 已提交并通过本地真实下载 | 当前按步骤先完成 DATA-04/05/06/07 | 后续在 CPU 服务器下载，不需要 GPU 或 819 GB 音频 |
| 3 数据 | DATA-09：WavCaps metadata/duration/leakage 审计 | TODO | `3c5b1e9` | 本地全量 403,050 条审计与确定性 hash 复核完成；89 项测试通过 | 等待远程 DATA-08；论文精确 post-blocklist 口径仍 `[MISSING]` | 远程生成 manifests/blocklists 并核对摘要 hash |
| 3 数据 | DATA-06：AudioCaps 2.0 官方 metadata 下载 | TODO | `4ef4f29` | 固定 commit、MD5/SHA256/bytes 与 CPU 下载脚本已提交；本地真实下载闭环通过 | 当前优先等待 DATA-04，尚未安排远程运行 | DATA-04 后在 CPU 服务器下载约 6.9 MB metadata |
| 3 数据 | DATA-07：AudioCaps 2.0 metadata/UIQ 全量校验 | TODO | `4ef4f29` | 本地真实全量验证通过：91,254 train、495 val、975 test、正/负 UIQ 精确对齐；82 项测试通过 | 远程尚未运行 | DATA-06 后生成远程 manifests 和统计证据 |
| 3 数据 | AudioCaps v2 论文 91,256 train 口径 | BLOCKED | `4ef4f29` | 官方固定 CSV 与公共 loader 已审计；另有 `[CODE]` 91,254 和 `[INFERRED]` 修复 91,254 两套 manifest | `[MISSING]` 能产生 91,256 个有效样本的论文 manifest/loader；实际音频尚未提供 | 请求官方音频；训练前显式选择公开 loader 或修复口径，不冒充 exact |
| 3 数据 | Clotho v2.1 evaluation 下载与 checksum | COMPLETED | `aff03e4` | DATA-01 首次下载及两次幂等复核完成；3 个 MD5 全部匹配 | 无；两次复核均为 `verified_existing`，未重复下载 | DATA-02 已完成安全解压、1,045 条 WAV/CSV/UIQ ID 校验 |
| 3 数据 | DATA-02：Clotho evaluation 解压与完整性校验 | COMPLETED | `7a8fb50` | `logs/data02_clotho_validation_20260717_233100`：1,045 WAV 全部解码；5 captions/clip；四类正 UIQ ID 精确对齐；manifest MD5 `253c1b275e3618fa94750150d7962da5` | 无；negative 仅验证了 `[INFERRED]` `.wav` 后缀映射，不含官方 target/HN 配对 | 保留小型审计摘要，待 CPU 数据准备完成后进入 GPU embedding |
| 3 数据 | DATA-03：Clotho development/validation 下载与 checksum | IN_PROGRESS | `f04128d` | 四个 CSV 已本地逐文件验证；最后已知远程下载目录 `logs/data03_clotho_trainval_download_20260716_170930`，当前主容器无下载进程 | 两个音频归档的远程完成状态尚未重新确认；六个文件精确总计 5,805,043,699 bytes | 不干扰当前 DATA-04；后续单独核对六个文件的 size/MD5/SHA256 |
| 3 数据 | DATA-10：Clotho development/validation 安全解压与全量校验 | TODO | `f04128d` | 本地四 CSV 全量审计和 97 项测试通过；远程 4,884 WAV 尚未解压/解码 | 等待 DATA-03 两个归档完成；论文 early-stopping split `[MISSING]` | DATA-04/05 后再安排 CPU 远程执行；公开代码与 clean-validation 路线分栏 |
| 3 数据 | DATA-04：固定并下载 MECAT `00A/test` | WAITING_USER | `87dbe32` | 独立工作副本 `/home/jg525/Omni-Embed-Audio-data04` 已创建且干净；下载命令尚未运行 | 需要远程 CPU 执行约 173 MB 下载 | 拉取最新分支后运行下载并核对 manifest/size/SHA256 |
| 3 数据 | DATA-05：MECAT 848 条安全解压/解码/UIQ ID 校验 | BLOCKED | `87dbe32` | 尚未运行；安全解压、FLAC 完整解码、六字段保留、UIQ 集合校验已通过合成测试 | 等待 DATA-04 | DATA-04 完成后在同一 CPU 服务器执行 |
| 3 数据 | MECAT 论文 847 条子集与检索 caption 口径 | BLOCKED | `87dbe32` | 官方 `00A/test` 与发布正向 UIQ 均为 848；审计文档已完成 | `[MISSING]` 被排除 ID、过滤规则、caption 字段/组合均未公开 | 不擅自删样本；先报告 848 条公开口径，继续请求作者证据 |
| 3 数据 | DATA-11：MECAT–WavCaps 来源视频候选审计 | IN_PROGRESS | `e5bf4db` | 本地发布 UIQ 848 IDs × 固定 AudioSet_SL 108,317 rows 已完成：807 唯一来源视频、4 个同源视频候选；109 项测试通过 | DATA-05 archive manifest 与 DATA-08 远程 metadata 尚未复算；论文音频/embedding 阈值 `[MISSING]` | DATA-05/08 后在 CPU 侧运行 canonical join；不把来源视频候选写成音频重复或 blocklist |
| 4 训练 | 首个 3B 过拟合/单卡/DDP/全量闭环 | BLOCKED | N/A | 未开始 | DDP、seed、早停、stage LR 等不完整 | 评测闭环后再补最小训练基础设施 |
| 5 基线 | 四个 CLAP 与三个 vanilla backbone 静态就绪度审计 | COMPLETED | `d13b115` | 干净 `d13b115` 上生成 `results/audits/baseline_readiness_d13b115.json`：7 个模型、5 个代码入口完整、3 个资源身份固定、0 个正式可运行、0 证据漂移；三种 vanilla 已具备固定协议、模型锁解析、base-hidden-dimension 可恢复生成器、强制 5-audio/25-query smoke→full 闸门及锁绑定 CPU 四协议 finalizer；真实锁和 GPU smoke 均尚未运行；完整 227 项测试通过 | 无；代码/锁/指标工具完成不代表基线已评测 | DATA-04 后按长任务规则逐个生成并提交小型 vanilla base 锁，再申请 1×A100-80GB 运行锁绑定 smoke；四个 CLAP 仍需固定 source/checkpoint |
| 5 基线 | LAION/Robust/MGA/M2D-CLAP | BLOCKED | N/A | 未开始 | checkpoint revision/统一入口不完整 | 官方 OEA 评测完成后逐个固定资源 |
| 6 扩展 | 人工/闭源 API、效率、token-length、派生图表 | TODO | N/A | 未开始 | 按阶段后置 | 核心训练评测完成后执行 |

## 当前焦点

- 当前阶段：Qwen3B-Cl/Clotho 官方权重 T2A/T2T 与四类正向 UIQ 已完成首个完整闭环；Qwen3B AudioCaps checkpoint 的全量 Clotho 首次尝试在模型加载前因门禁入口导入失败，失败证据和最小修复均已提交，等待同 RUN_ID 重试。
- 当前对应论文范围：所有主表 1–5、附录表 6–17、图 1–3、附录 A–M 已建清单。
- 最近完成：`f7ac255` 修复 `python scripts/verify_oea_smoke_gate.py` 直接入口缺少仓库根路径的问题；新增实际子进程入口测试并固化 exit 1 失败记录，完整 243 项测试通过。该失败未加载模型、未生成 embedding、未改变任何评测口径。
- 当前阻塞：Qwen3B-Cl/Clotho 当前闭环无阻塞；完整三数据集/多模型表仍缺 AudioCaps/MECAT 音频、部分模型资源和若干论文未公开口径。训练阶段按用户要求暂停。
- 下一步：远程 RTX 4090 拉取 `f7ac255` 后复用 RUN_ID `oea_qwen3b_ac_clotho_embeddings_seed42_20260719_231724`，让 wrapper 新建独立 attempt 并完成 1,045 个 Clotho 音频和 5,225 个 caption embedding。
