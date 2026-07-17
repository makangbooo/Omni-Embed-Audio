# Omni-Embed-Audio 复现状态

最后更新：2026-07-18（Asia/Shanghai）

状态值仅使用：`TODO`、`IN_PROGRESS`、`WAITING_USER`、`RUNNING_REMOTE`、`COMPLETED`、`FAILED`、`BLOCKED`。

| 阶段 | 任务 | 状态 | 当前 Commit | 远程状态 | 阻塞原因 | 下一步 |
|---|---|---|---|---|---|---|
| 0 项目审计 | 完整阅读论文正文/附录/表/图/脚注 | COMPLETED | `48d7a50` | fork 已同步 | 无 | 固定 inventory |
| 0 项目审计 | 审计 README、依赖、训练/评测、数据、UIQ、HN、指标 | COMPLETED | `48d7a50` | fork 已同步 | 无 | 用户确认审计结论 |
| 0 项目审计 | 31 条实验/图/派生分析 inventory | COMPLETED | `48d7a50` | fork 已同步 | 无 | 用户确认复现边界 |
| 0 GitHub | 建立 `repro/oea-full` 分支 | COMPLETED | `48d7a50` | 本地与 fork 分支均存在 | 无 | 后续提交仅推送该分支 |
| 0 GitHub | commit 并 push 第一阶段审计 | COMPLETED | `48d7a502e279bd9a2a3195352163626fb1781a3b` | `origin/repro/oea-full` 已建立 | 无 | 保持本地与远程实验 commit 一致 |
| 1 环境 | CPU-01：CPU/内存/存储/编译器/网络审计 | COMPLETED | `20770d0` | 远程完成；原始日志未进 Git | Zenodo 返回 503；容器 `nproc=2` | 在共享目录挂载到 A100 后执行 GPU-01 |
| 1 环境 | GPU-01：A100/Driver/CUDA/NCCL 基础审计 | COMPLETED | `905e84d` | 1×A100-SXM4-80GB 远程完成；原始日志未进 Git | 多卡 GPU–GPU 拓扑留待 DDP smoke | 固定环境候选并执行 GPU-02 |
| 1 环境 | 生成环境候选、安装脚本与无模型检查 | COMPLETED | `c61fecb` | fork 已同步 | 完整 transitive lock 必须在 Linux solve 后生成 | 固定首批模型资源 |
| 1 环境 | CPU-02：创建 `oea-repro` 并生成 resolved lock | COMPLETED | `11f20d2` | `pip check`、必需接口、仓库导入、音频 I/O/重采样均通过；resolved artifacts 已生成 | 无 | 在单卡实例执行 GPU-02 |
| 1 环境 | GPU-02：CUDA/BF16/NCCL 验证 | COMPLETED | `711ef28` | 1×A100-SXM4-80GB 验证通过；原始日志保存在共享目录 | 无；`flash-attn` 为非必需可选项 | 下载首批固定 revision 模型资源 |
| 1 环境 | CPU-03：安装 DATA-02 的 7-Zip 工具 | COMPLETED | `a54cf5e` | `7zip=26.02` 已在共享 `oea-repro` 可用；install/wrapper exit 均为 0 | 无；Conda 报告目标包已安装，未发生依赖 transaction | 启动并行资源下载 |
| 2 官方权重 | MODEL-01：Qwen2.5-Omni-3B + OEA-Qwen3B-Cl | COMPLETED | `bba6084` | 断点续传完成；两个 asset 和全部 LFS SHA256 通过；无临时文件/下载进程 | 无；首次失败证据保留在旧运行目录 | 审计 checkpoint 内部结构 |
| 2 官方权重 | MODEL-02：OEA-Qwen3B-AC checkpoint | RUNNING_REMOTE | `ae2c3fb` | PID 52680；`logs/model02_download_20260716_171806`；3.42 GB 分片在更新 | CAS 间歇返回 504，但断点续传有实际增长 | 保持 tmux，完成后核对 9,466,835,918 bytes 与 LFS SHA256 |
| 2 官方权重 | MODEL-03：Nemotron-3B base + OEA-Nemo3B AC/Cl | RUNNING_REMOTE | `48c9b3c` | `logs/model03_download_20260717_142429`；base/AC 完整，Cl 分片 7,224,688,640/9,466,834,755 bytes 且在增长 | CAS 链路不稳定；单 worker、20 次重试正在缓解 | 保持 tmux；结束后核对 exit code、manifest、残留 partial 与 LFS SHA256 |
| 2 官方权重 | MODEL-04：Qwen2.5-Omni-7B base + OEA-Qwen7B AC/Cl | RUNNING_REMOTE | `48c9b3c` | `logs/model04_download_20260717_143427`；base 完整，AC 分片 11,146,362,880/17,940,610,993 bytes；Cl 尚未开始 | CAS 链路不稳定；AC 已进入第 2 次外层尝试 | 保持 tmux；AC 完成后自动下载 Cl，最后核对全部 SHA256 |
| 2 官方权重 | 5 个 Clotho 样例 Qwen3B-Cl smoke | COMPLETED | `fba8c3c` | 运行 `..._20260716_130619` 严格离线成功；5 音频/5 查询、544 LoRA、双 head、512 维归一化 embedding 全部通过；峰值 9.141 GiB allocated | 无；首次失败 `..._20260716_125636` 仍保留 | 固定小型结果摘要并进入检索指标单元测试 |
| 2 官方权重 | Tables 2/3：单个 3B 官方权重 T2A/T2T | TODO | `7a8fb50` | Clotho 1,045 条 evaluation manifest 已完整验证；GPU 正式 embedding 尚未安排 | 完整 AudioCaps/MECAT 候选集尚未准备；全三数据集表仍不能运行 | CPU 数据步骤完成后申请 1×A100-80GB，先做 Qwen3B-Cl 小批校验再生成全量 embedding |
| 2 官方权重 | Qwen3B-Cl 全量 checkpoint embedding 生成器 | COMPLETED | `03850ff` | 固定 1,045 audio/5,225 caption、单批原子工件、SHA256、严格离线/单 GPU、精确 resume identity 和失败 attempt 已实现；完整 67 项测试通过 | GPU 正式运行尚未开始；论文 `passage:` 与公开代码 no-prefix 冲突已显式标记 | CPU 数据步骤完成后先做少量 chunk GPU 校验，再启动可恢复全量生成 |
| 2 指标 | Canonical T2A/T2T/UIQ embedding evaluator | COMPLETED | `43158ae` | 确定性 ID 检索、caption 多正例/排除 self、显式 query 子集、完整排名和严格输入校验已实现；相关 15 项测试通过 | 论文未公开 T2T caption 选择与 tie 口径；已在协议文档标为 `[MISSING]` | 接入 checkpoint embedding 生成器 |
| 2 指标 | 可审计 embedding 实验输出 runner | COMPLETED | `5ed57a7` | CPU-only runner 已保存固定 embedding、metadata、完整排名、输入/工件 SHA256、seed、Git/环境/命令和失败证据；相关测试总计 27 项通过 | 无；正式运行要求干净 worktree 和显式协议来源 | 用 Qwen3B-Cl + Clotho evaluation 完成首个正式 T2A/T2T 闭环 |
| 2 指标 | Figure 3 / Table 17 指标单元测试 | COMPLETED | `88a7d1f` | R@k、Δ-Rank、HNSR、HNSR@k、TFR、TFR-HN@k 已按论文公式实现；7 个合成测试及完整 24 项测试通过 | 无；正式表 17 仍缺 target-HN audio ID | 在 canonical negative evaluator 中接入已验证指标 |
| 2 UIQ | Tables 12–15 正向 UIQ | IN_PROGRESS | `7a8fb50` | schema adapter 与 Clotho 1,045 条音频/四类正 UIQ 精确对齐已完成 | AudioCaps/MECAT 音频候选集未准备；MECAT 847/848 口径未决 | 数据就绪后接入 canonical ID evaluator，禁止静默丢弃未映射 ID |
| 2 Negative | Tables 4/17 否定查询 | BLOCKED | N/A | 未开始 | 发布文件无 HN audio ID | 请求作者 pairing 或审计式重建 |
| 3 数据 | WavCaps duration + leakage blocklists | IN_PROGRESS | `3c5b1e9` | 本地两次全量 metadata pass 已完成；精确复现 173 个 AudioCaps 与 638 个 Clotho 文件名重叠 | 论文 `<=31s` 与公开元数据计数冲突；精确 filtered manifest、Clotho 消歧和论文 blocklist 未发布 | 远程执行 DATA-08/09 复算；训练前再准备音频 |
| 3 数据 | DATA-08：固定 WavCaps metadata 下载 | TODO | `3c5b1e9` | 8 个固定文件、176,863,095 bytes、逐文件 SHA256/LFS ID 已提交并通过本地真实下载 | 当前按步骤先完成 DATA-04/05/06/07 | 后续在 CPU 服务器下载，不需要 GPU 或 819 GB 音频 |
| 3 数据 | DATA-09：WavCaps metadata/duration/leakage 审计 | TODO | `3c5b1e9` | 本地全量 403,050 条审计与确定性 hash 复核完成；89 项测试通过 | 等待远程 DATA-08；论文精确 post-blocklist 口径仍 `[MISSING]` | 远程生成 manifests/blocklists 并核对摘要 hash |
| 3 数据 | DATA-06：AudioCaps 2.0 官方 metadata 下载 | TODO | `4ef4f29` | 固定 commit、MD5/SHA256/bytes 与 CPU 下载脚本已提交；本地真实下载闭环通过 | 当前优先等待 DATA-04，尚未安排远程运行 | DATA-04 后在 CPU 服务器下载约 6.9 MB metadata |
| 3 数据 | DATA-07：AudioCaps 2.0 metadata/UIQ 全量校验 | TODO | `4ef4f29` | 本地真实全量验证通过：91,254 train、495 val、975 test、正/负 UIQ 精确对齐；82 项测试通过 | 远程尚未运行 | DATA-06 后生成远程 manifests 和统计证据 |
| 3 数据 | AudioCaps v2 论文 91,256 train 口径 | BLOCKED | `4ef4f29` | 官方固定 CSV 与公共 loader 已审计；另有 `[CODE]` 91,254 和 `[INFERRED]` 修复 91,254 两套 manifest | `[MISSING]` 能产生 91,256 个有效样本的论文 manifest/loader；实际音频尚未提供 | 请求官方音频；训练前显式选择公开 loader 或修复口径，不冒充 exact |
| 3 数据 | Clotho v2.1 evaluation 下载与 checksum | COMPLETED | `aff03e4` | DATA-01 首次下载及两次幂等复核完成；3 个 MD5 全部匹配 | 无；两次复核均为 `verified_existing`，未重复下载 | DATA-02 已完成安全解压、1,045 条 WAV/CSV/UIQ ID 校验 |
| 3 数据 | DATA-02：Clotho evaluation 解压与完整性校验 | COMPLETED | `7a8fb50` | `logs/data02_clotho_validation_20260717_233100`：1,045 WAV 全部解码；5 captions/clip；四类正 UIQ ID 精确对齐；manifest MD5 `253c1b275e3618fa94750150d7962da5` | 无；negative 仅验证了 `[INFERRED]` `.wav` 后缀映射，不含官方 target/HN 配对 | 保留小型审计摘要，待 CPU 数据准备完成后进入 GPU embedding |
| 3 数据 | DATA-03：Clotho development/validation 下载与 checksum | IN_PROGRESS | `ae2c3fb` | 最后已知运行目录 `logs/data03_clotho_trainval_download_20260716_170930`；当前主容器未看到下载进程，但不能据此判断另一实例状态 | 跨实例进程状态尚未重新确认；六个 Zenodo 文件约 5.8 GB `[INFERRED]` | 不干扰当前 DATA-04；后续单独核对逐文件 MD5 与下载进程 |
| 3 数据 | DATA-04：固定并下载 MECAT `00A/test` | WAITING_USER | `87dbe32` | 独立工作副本 `/home/jg525/Omni-Embed-Audio-data04` 已创建且干净；下载命令尚未运行 | 需要远程 CPU 执行约 173 MB 下载 | 拉取最新分支后运行下载并核对 manifest/size/SHA256 |
| 3 数据 | DATA-05：MECAT 848 条安全解压/解码/UIQ ID 校验 | BLOCKED | `87dbe32` | 尚未运行；安全解压、FLAC 完整解码、六字段保留、UIQ 集合校验已通过合成测试 | 等待 DATA-04 | DATA-04 完成后在同一 CPU 服务器执行 |
| 3 数据 | MECAT 论文 847 条子集与检索 caption 口径 | BLOCKED | `87dbe32` | 官方 `00A/test` 与发布正向 UIQ 均为 848；审计文档已完成 | `[MISSING]` 被排除 ID、过滤规则、caption 字段/组合均未公开 | 不擅自删样本；先报告 848 条公开口径，继续请求作者证据 |
| 4 训练 | 首个 3B 过拟合/单卡/DDP/全量闭环 | BLOCKED | N/A | 未开始 | DDP、seed、早停、stage LR 等不完整 | 评测闭环后再补最小训练基础设施 |
| 5 基线 | LAION/Robust/MGA/M2D-CLAP | BLOCKED | N/A | 未开始 | checkpoint revision/统一入口不完整 | 官方 OEA 评测完成后逐个固定资源 |
| 6 扩展 | 人工/闭源 API、效率、token-length、派生图表 | TODO | N/A | 未开始 | 按阶段后置 | 核心训练评测完成后执行 |

## 当前焦点

- 当前阶段：环境、首批官方权重、checkpoint 结构审计、inference-only 权重提取及单卡 GPU smoke 已完成。
- 当前对应论文范围：所有主表 1–5、附录表 6–17、图 1–3、附录 A–M 已建清单。
- 最近完成：Clotho DATA-02 远程全量解码/manifest/UIQ 校验通过；WavCaps DATA-09 在固定公开元数据上完成两次全量审计，精确复现论文的 173 个 AudioCaps 和 638 个 Clotho 重叠计数；完整 89 项测试通过。
- 当前阻塞：正式论文全三数据集表仍缺 AudioCaps 与 MECAT 音频；AudioCaps 还缺论文有效 91,256-row train manifest，MECAT 还缺论文 847 条中的排除 ID和检索 caption 字段。WavCaps 的 `<=31s` 文字条件与 275,618 计数不一致，论文精确 filtered manifest/blocklist 未发布。当前 5 条样例只证明首个官方 checkpoint 的最小前向闭环，不代表论文 Recall 复现。
- 下一步：保持 DATA-04 为用户唯一待执行步骤；成功后在同一 CPU 服务器执行 DATA-05，再依次完成 DATA-06/07 与 DATA-08/09。CPU 数据步骤完成后再申请 1×A100-80GB 做 Qwen3B-Cl 正式 embedding 小批校验。
