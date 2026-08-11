# Omni-Embed-Audio 复现项目说明

最后整理：2026-08-11（Asia/Shanghai）

## 1. 当前范围

本仓库只保留 Omni-Embed-Audio（OEA）论文复现所需的代码、配置、测试、
小型审计结果和说明文档。当前主实验是使用作者发布的官方 checkpoint 做推理
和评测；没有完成按论文配置从头训练 OEA，因此不能把本项目描述成“自行训练
并复现了 OEA”。

保留范围：

- 六个 OEA 官方变体的 checkpoint 固定、加载、embedding 生成和评测；
- Clotho、AudioCaps、MECAT、WavCaps/UIQ 的数据准备与审计；
- Text-to-Audio（T2A）、Text-to-Text（T2T）、正向 UIQ 和 Negative UIQ；
- 论文 Tables 2–17、Figure 2 及相关派生表的可审计结果；
- LAION-CLAP、M2D-CLAP、MGA-CLAP、Robust-CLAP 和三种 vanilla backbone；
- RTX 4090/A100 上的受控效率测量；
- 环境、模型锁、输入输出哈希、失败边界和复现测试。

所有在 OEA 之上开展的后续方法探索已经从当前代码树删除。它们不再是当前
项目的依赖、待办或结果组成。需要追溯时只能查看 Git 历史，不能从当前
`HEAD` 执行。

## 2. 结果性质

本项目严格区分以下状态：

| 状态 | 含义 |
|---|---|
| `STRICT` | 论文协议、资源身份和输入输出均有充分证据 |
| `CONTROLLED` | 使用公开资源和预先声明的替代协议完成运行 |
| `CLOSE` | 数值接近论文，但仍存在未公开的协议差异 |
| `BLOCKED` | 缺少作者数据、配对或评测细节，不能严格复现 |

论文实验 inventory 的最后一次整行审计是 `4/31 = 12.9%` 严格完成。这个
比例很低，是因为一行只有在全部模型、数据和论文协议都闭环后才算完成。
后续 public-controlled 覆盖明显更广，但不能反向改写成 strict reproduction。
详细矩阵见 `results/tables/paper_experiment_matrix.csv` 和
`results/tables/oea_partial_tables.md`。

## 3. OEA 方法与 checkpoint

OEA 用同一个多模态大模型接收文本 query 和音频 passage，分别经过文本/音频
projection head，输出 512 维归一化 embedding，以点积完成检索：

```text
text query  -> shared MLLM -> text projection  -> 512-d embedding
audio       -> shared MLLM -> audio projection -> 512-d embedding
                                      cosine/dot-product ranking
```

代码审计确认：最后层 hidden state 按 attention mask mean-pooling；projection
为 `Linear(no bias) -> Dropout -> LayerNorm -> L2 normalize`；官方训练冻结
backbone，仅训练 LoRA 和双 projection heads，并使用对称 InfoNCE。公开训练
数据是 audio-caption 配对，不是 spoken-query 数据。

本项目固定的六个官方变体是：

| Backbone | AudioCaps 版 | AudioCaps + Clotho 版 |
|---|---|---|
| Nemotron-3B | OEA-Nemo3B-AC | OEA-Nemo3B-Cl |
| Qwen2.5-Omni-3B | OEA-Qwen3B-AC | OEA-Qwen3B-Cl |
| Qwen2.5-Omni-7B | OEA-Qwen7B-AC | OEA-Qwen7B-Cl |

模型 revision、文件身份和推理配置位于 `configs/checkpoints/`、
`configs/eval/` 与 `results/model_locks/`。模型大权重不提交到 Git，保存在
远程 `/home/jg525/models/oea`。

## 4. 复现阶段

### 4.1 论文与代码审计

逐项阅读论文正文、附录、表格、图、README、训练代码、评测代码和 UIQ
生成代码；建立 31 行论文实验 inventory，并转录约 910 个论文指标。所有
协议分别标为 `[PAPER]`、`[CODE]`、`[INFERRED]` 或 `[MISSING]`，避免根据
结果接近程度事后选择协议。

### 4.2 环境与运行治理

建立 `oea-repro` Conda 环境、resolved dependency lock、CPU/GPU preflight、
BF16/CUDA 检查和统一运行入口。长任务通过普通 tmux 会话执行，每个运行保存
Git commit、命令、退出码、stderr、manifest、metrics 和 SHA256。

### 4.3 官方资源与 checkpoint

对 base model 和 OEA checkpoint 做 revision、size、LFS/SHA256、结构和
loadability 审计。正式评测先运行小样本 smoke，再分块生成全量 embedding，
最后在 CPU 上计算排名。主实验始终使用官方 checkpoint，没有为追求论文
数值而选择其他 checkpoint。

### 4.4 数据准备

- Clotho evaluation：1,045 个音频，每个 5 captions；完成归档校验、解码和
  UIQ ID 对齐。
- AudioCaps test：975 个音频、4,875 captions；公开 train manifest 是
  91,254 行，而论文写 91,256，差异保留为 `[MISSING]`。
- MECAT public：848 个音频全部解码；论文写 847，未公开排除 ID，因此严格
  847-row 协议保持 blocked。
- WavCaps：完成 metadata、时长、来源和与评测集潜在重叠审计。
- UIQ：正向四类共 11,472 条，Negative 共 1,581 条。

### 4.5 主表与基线

Clotho、AudioCaps 和 MECAT public-848 的 public-controlled Table 2/3 矩阵
已经运行。覆盖 OEA 六变体、四个 CLAP 系基线和三个 vanilla backbone。
正向 UIQ 在三个数据集上的十模型受控矩阵也已完成。

代表性结果：

| 模型/协议 | R@1 | R@5 | R@10 | 判定 |
|---|---:|---:|---:|---|
| OEA-Nemo3B-Cl / Clotho T2A all-caption | 21.7225 | 47.1196 | 60.4402 | `CONTROLLED/CLOSE` |
| OEA-Nemo3B-Cl / Clotho T2T seed-0 | 62.9665 | 75.9809 | 80.8612 | `CONTROLLED/CLOSE` |
| vanilla Nemotron-3B / Clotho T2A all-caption | 7.2536 | 21.3014 | 30.1818 | `CONTROLLED/CLOSE` |
| LAION-CLAP / Clotho T2A all-caption | 14.0478 | 37.2057 | 49.8182 | `CONTROLLED` |

严格 Table 2/3 仍受 caption selection、T2T self-exclusion/tie 口径、
MECAT 847-row manifest 和 audio `passage:` 前缀差异限制。

### 4.6 Negative UIQ 与 Tables 4/17

公开 Negative UIQ 文件没有完整发布 target/hard-negative 音频 ID 配对。本
项目实现了 deterministic caption-identity pairing audit，禁止通过 embedding
最近邻猜测论文配对。在该公开可重建协议下，三数据集、十模型的 Negative
UIQ controlled evaluation 已完成，1,581/1,581 条查询均进入配对审计。

因此可以声明“Table 4/17 的公开可重建受控实验已完成”，不能声明“作者未
公开的 exact target/HN pairing 已严格复现”。核心证据：

- `results/audits/negative_uiq_exact_pairing_audit_20260803.json`
- `results/audits/oea_negative_uiq_eval_20260803.json`
- `results/audits/clap_negative_uiq_eval_20260803.json`
- `results/audits/m2d_clap_negative_uiq_eval_20260803.json`

### 4.7 效率实验

完成 RTX 4090 上的受控 latency、throughput、peak memory 和参数量测量，并
保存 A100 对照边界。论文没有公开完整 timing/warmup/batch/memory counting
口径，因此 Tables 5/16 的同硬件 strict claim 仍受限。

## 5. 主要困难与处理

| 问题 | 处理方式 |
|---|---|
| tmux 会话退出、重复会话、日志粘贴不完整 | 使用普通 tmux；以 metrics、manifest、RC 和 SHA256 为完成证据 |
| `/home` 与 `/file_system` 路径不同 | 命令先进入仓库根目录；结果同时记录绝对路径和文件身份 |
| `ModuleNotFoundError: AudioRetrieval` | 固定 cwd/Conda 环境，并让脚本显式加入仓库根目录 |
| checkpoint 名称和 README 不一致 | 固定 revision、真实文件名、size 和 LFS/SHA256，不凭示例名推测 |
| PyTorch 安全加载及完整 base tensor 混入 LoRA state | 使用安全结构检查、非覆盖提取和逐 tensor 验证 |
| AudioCaps 91,256/91,254、MECAT 847/848 | 分栏报告论文与公开数据口径，不补造样本 |
| caption selection 与 T2T tie 未公开 | 预先声明多个协议并全部报告，不按接近论文程度择优 |
| Negative UIQ 缺 exact 音频配对 | 使用 caption identity 受控配对并明确 claim boundary |
| 长任务或大文件 hash 中断 | 分块、可恢复输出、原子完成标记和失败目录保留 |

## 6. 当前可复现程度

| 范围 | 当前状态 |
|---|---|
| 官方 checkpoint 推理 | 已完成主要六变体资源固定和评测闭环 |
| Clotho/AudioCaps/MECAT public 主表 | public-controlled 矩阵已完成；strict 协议仍有缺口 |
| 正向 UIQ | 三数据集十模型受控矩阵完成 |
| Negative UIQ / Tables 4/17 | 三数据集十模型受控矩阵完成；exact pairing blocked |
| CLAP 与 vanilla 基线 | 受控矩阵完成 |
| 效率 | 4090 受控测量完成；论文未公开口径导致 strict claim 受限 |
| OEA 从头训练 | 未复现 |

## 7. 证据与目录

- `configs/checkpoints/`：官方资源固定信息。
- `configs/eval/`：正式评测配置。
- `configs/reproduction/`：复现阶段注册表。
- `results/model_locks/`：模型与 checkpoint 身份。
- `results/audits/`：小型、可提交的运行证据。
- `results/observations/`：论文指标与复现观测绑定。
- `results/tables/`：论文矩阵、部分表和派生表。
- `scripts/`：数据、embedding、评测、基线和结果汇总入口。
- `docs/`：协议、数据审计和复现说明。

少数必须保留的审计仍带早期 `asrur_` 前缀，但内容属于 OEA 论文复现：
`asrur_nemo_clotho_t2a_20260727.json`、
`asrur_nemo_g1_full_embeddings_20260727.json`、
`asrur_nemo_g1_smoke_20260727.json`、
`asrur_nemo_phase1_success_20260727.json` 和
`asrur_vanilla_nemo_portable_lock_20260727.json`。这些文件支撑 T2A/T2T、
Figure 2 或模型锁，不能按文件名前缀删除。

## 8. 重新执行顺序

1. 阅读 `README_REPRODUCTION.md`、本文件和 `docs/evaluation_protocol.md`。
2. 创建 `oea-repro` 环境并运行 `scripts/check_environment.py`。
3. 按 `docs/official_checkpoint_preparation.md` 准备官方 checkpoint 和 model lock。
4. 按各数据审计文档准备 Clotho、AudioCaps、MECAT 和 WavCaps/UIQ。
5. 先运行模型/数据 smoke，再生成完整 embedding。
6. 运行 T2A/T2T、正向 UIQ、Negative UIQ 和基线矩阵。
7. 用 `scripts/build_reproduction_summary.py` 和
   `scripts/build_paper_experiment_matrix.py` 重新生成/核对小型结果。

当前仓库的目标是让这条官方论文复现链条保持清楚、可检查、可再次执行，
不再承担任何后续创新实验。

## 9. 最终最小化清理

2026-08-12 的第二轮清理删除了两类不参与正式实验执行的文件：一类是只对
已提交结果做重复断言的 evidence 测试；另一类是早期资源 presence/probe、
服务器接管和 tmux 调度包装。相同实验的正式数据、模型锁、embedding、评测
和汇总入口均保留。

没有删除全部测试。核心指标、UIQ schema、模型锁、checkpoint 安全、数据
校验、embedding 生成/评测和统一复现入口的测试仍保留，用于防止清理破坏
主流程。也没有按名称批量删除 `validate_*` 或 `audit_*`：凡是被数据准备、
正式 GPU wrapper、基线 runner 或结果汇总调用的校验器都属于复现依赖。

历史 `logs/`、`results/audits/` 和主实验 `results/raw/` 继续保留。它们记录
成功运行、失败重试、输入哈希、退出码和协议边界，是复现过程证据而不是
待清理的创新实验产物。
