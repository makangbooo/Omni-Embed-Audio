# ASR-Uncertainty Reranking：Phase 0 工作区审计

审计日期：2026-07-26（Asia/Shanghai）；最新只读复核：2026-07-26 16:41

本文件记录新项目
“ASR-Uncertainty-Guided Reranking over OEA for Robust Spoken Query Retrieval”
开始前的只读状态。Phase 0 未下载数据、模型或依赖，未启动 GPU，也未训练任何模型。

## 1. 审计边界与证据等级

- `LOCAL_VERIFIED`：在当前 Windows/Codex 工作区直接读取或执行得到。
- `COMMITTED_EVIDENCE`：由当前 Git 仓库内的固定配置、测试或小型审计结果支持。
- `REMOTE_EVIDENCE`：由用户提供的远程终端输出支持；不等价于 2026-07-26 的实时文件检查。
- `NOT_DETECTED`：当前仓库和既有远程输出均没有可审计的存在证据；不是对远程全部磁盘的否定断言。
- `PENDING_REMOTE`：远程任务尚在运行，不能把预期结果当成已完成事实。

现有文件、结果、模型和缓存均视为用户所有。审计没有覆盖、删除或移动任何用户资产。

## 2. Git 与目录结构

| 项目 | 实际状态 | 证据 |
|---|---|---|
| 仓库根目录 | `C:\Users\jigaiii\Documents\Codex\2026-07-15\omni-embed-audio-leveraging-multimodal-llms-3\Omni-Embed-Audio` | `LOCAL_VERIFIED` |
| 分支 | `repro/oea-full` | `LOCAL_VERIFIED` |
| Phase 0 审计前 HEAD | `8d98c25636f4e34b7ee1c394a0894ea49d1eb46b` | `LOCAL_VERIFIED` |
| 最新只读复核前 HEAD | `a851a2b40d607f5ca38722b93406b5d885484b18` | `LOCAL_VERIFIED` |
| 工作树 | 最新复核开始时干净 | `LOCAL_VERIFIED` |
| fork | `https://github.com/makangbooo/Omni-Embed-Audio.git` | `LOCAL_VERIFIED` |
| upstream | `https://github.com/JudeJiwoo/Omni-Embed-Audio.git` | `LOCAL_VERIFIED` |

当前项目已经有合理目录结构，不机械创建第二套空目录：

| 新项目逻辑目录 | 复用位置 | Phase 0 状态 |
|---|---|---|
| `src/` | `AudioRetrieval/asr_uncertainty_reranking/` | 已创建并包含 normalization、aggregation、metrics、cache manifest |
| `configs/` | `configs/asr_uncertainty_reranking/` | 已创建固定资源清单；正式模型 runner 配置待 Phase 1/2 |
| `scripts/` | 根目录已有 `scripts/` | 已存在 |
| `tests/` | 根目录已有 `tests/` | 已存在 |
| `results/` | `results/raw/`（Git 忽略）及 `results/audits/`、`results/tables/` | 已存在 |
| `logs/` | 根目录 `logs/`，已被 `.gitignore` 排除 | 远程运行时创建 |
| `cache/` | `/home/jg525/model_cache/` 与 `/home/jg525/datasets/oea/` | 使用仓库外共享存储 |

最新文件规模复核：`AudioRetrieval/` 204 个文件、`configs/` 44 个文件、
`data/` 17 个文件、`docs/` 23 个文件、`results/` 33 个文件、`scripts/` 189
个文件、`tests/` 154 个文件。根目录没有已跟踪的 `cache/`、`logs/` 或 `src/`，这是既有
隔离策略，不是资源缺失。

## 3. OEA、SQuTR 与评测代码

### 3.1 已有 OEA 能力

`COMMITTED_EVIDENCE`：

- 官方 OEA 仓库代码已在当前 fork 中。
- `AudioRetrieval/preprocessing/embeddings/oea.py` 可加载 LoRA、音频/文本
  projection heads，并输出 512 维向量。
- `AudioRetrieval/models/omni_embed_adapter.py` 使用 attention-mask mean pooling
  和 L2 normalization。
- 已有官方资源审计、inference-only checkpoint 提取、model lock、embedding
  生成、Clotho 评测、失败证据保存和缓存哈希工具。
- `scripts/evaluate_frozen_text_index.py` 已支持冻结文档 embedding、graded qrels、
  `nDCG@10`、`MRR@10`、Recall 与逐查询排名证据。

### 3.2 已有 SQuTR 能力

`COMMITTED_EVIDENCE`：

- 当前仓库没有 vendored SQuTR 上游源码；已有固定上游 commit
  `cc3fb31fc0dc44fef3a44c569b344516bbaee79c` 的协议审计。
- 已有 SQuTR 下载、ZIP 路径安全审计、可恢复提取、CRC、JSONL schema、
  qrels/ID 闭包和 WAV 解码验证工具。
- DATA-13A 的已提交小型证据位于
  `results/audits/squtr_data13a_archive_audit_20260726.json`。

SQuTR 官方实现不能直接作为本项目最终 OEA/ASR 模块：

1. `src/retrieval/omni_emb.py` 直接加载
   `nvidia/omni-embed-nemotron-3b`，只对 backbone 最后一层做 mean pooling 和
   L2 normalization，没有加载 OEA LoRA 和 512 维模态 projection heads。
2. SQuTR README 把 `queries_with_audio_*.jsonl` 放在子集根目录；官方
   `CustomAudioRetrieval` 却把 `query_file` 拼到 `audio_path`。真实 ZIP 的
   DATA-13A 审计确认 README 的根目录结构才与归档一致。
3. SQuTR Whisper 脚本使用 temperature 0，只读取 `output.outputs[0].text`，
   因而只提供 1-best，不提供本项目所需的 4-best、sequence/token score 或
   不确定性特征。

这些差异要求复用数据协议和可比基线，而不是无审计地复用上游 runner。

## 4. 数据资产

### 4.1 SQuTR

| 项目 | 状态 | 实际证据 |
|---|---|---|
| 固定版本 | 完成 | HF revision `2f1b041e2e98e0d28ed68fbcf22126ef247eb719` |
| 许可证 | 已记录 | CC BY-SA 4.0 |
| ZIP | 完成 | `/home/jg525/datasets/oea/squtr/source/source_data.zip` |
| ZIP 大小 | 完成 | 21,069,841,248 bytes |
| ZIP SHA256 | 完成 | `8956bf938de3f9ce168a1e7daf2ff61b0b7fe603fa5c3d7dc6a4314617c6997c` |
| 结构审计 | 完成 | 149,349 records、149,268 WAV、42 JSONL、零路径安全违规 |
| 解压后成员大小 | 已审计 | 28,422,366,590 bytes |
| FiQA test | 归档中存在 | 648 queries × 4 声学条件 |
| NQ test | 归档中存在 | 3,452 queries × 4 声学条件 |
| 全量内容校验 | `WAITING_USER` | extraction/CRC 已完成；DATA-13C attempt 3 因并发锁申请失败在校验前退出，final manifest 尚未生成 |

DATA-13B 已在 `/home/jg525/datasets/oea/squtr/extracted` 完成安全解压和全量
CRC。后续两次内容校验分别暴露了官方空 corpus 行和字符串 qrels score，代码已按
固定 SQuTR loader 语义做最小修复。第三次恢复在进入校验前因
`.data13c_content_recovery.lock` 申请失败退出，wrapper code 为 `20`；v2 probe
cache 与 `/home/jg525/datasets/oea/squtr/manifests/squtr_audio_query_manifest.jsonl`
均未生成。在锁状态只读审计以及三个退出码、实际 schema、qrels 分布、候选数和
全部 WAV 验证完成前，不生成 SQuTR embedding。

### 4.2 FiQA 与 NQ

| 资源 | 目前可确认的状态 |
|---|---|
| FiQA test corpus/query/qrels | SQuTR ZIP 中存在；内容级校验正在 DATA-13B 运行 |
| FiQA train/dev | `NOT_DETECTED`；没有已下载的独立 `mteb/fiqa` 证据 |
| NQ test corpus/query/qrels | SQuTR ZIP 中存在；内容级校验正在 DATA-13B 运行 |
| 独立 `mteb/nq` snapshot | `NOT_DETECTED`，且当前不是必须下载项 |

官方元数据固定候选：

- `mteb/fiqa` revision `5e59eeb3a7df6b85882112b747008547c21587ea`，
  仓库总文件大小 48,881,656 bytes，license 字段为 `unknown`。
- `mteb/nq` revision `b84726e65fd226125cf7c0cbeeb5c214d49e8187`，
  仓库总文件大小 1,461,500,309 bytes，CC BY-NC-SA 3.0。

用户给出的 FiQA 5,500/500/648 和 NQ 3,452 规模尚未由本地文件重新计数；
DATA-13B 只验证 SQuTR test。FiQA train/dev 下载后必须独立验证 split 数量、
ID/文本重叠和 qrels 闭包。

## 5. 模型缓存

以下“完成”是 2026-07-17 远程下载日志的 LFS SHA256 完成证据；在首次模型运行前
仍需一次只读实时复核，防止用户之后移动或删除文件。

| 模型/检查点 | 固定 revision | 状态 | 远程证据 |
|---|---|---|---|
| `nvidia/omni-embed-nemotron-3b` | `865db1bb57e369a85357cf114cbd6b3c5322d19d` | `REMOTE_EVIDENCE: complete` | 22 files、9,423,120,401 bytes、目录约 8.8 GiB |
| `JudeJiwoo/OEA-Nemo3B-AC` | `8ed66aa77bc6f2001b807b5d2bd3e60503d89535` | `REMOTE_EVIDENCE: complete` | 3 files、9,466,834,697 bytes |
| `JudeJiwoo/OEA-Nemo3B-Cl` | `9588912298afca0b11f5895b864ae28083f35022` | `REMOTE_EVIDENCE: complete` | 3 files、9,466,834,755 bytes |
| `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | `NOT_DETECTED` | 无已有缓存证据 |
| `BAAI/bge-base-en-v1.5` | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | `NOT_DETECTED` | 无已有缓存证据 |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | `NOT_DETECTED` | 无已有缓存证据 |

OEA README 声称每个 checkpoint repo 含 `step_40.pt`，但固定的 Nemo 仓库实际
文件分别是 `step_400_best.pt` 和 `step_450_best.pt`。本项目以 immutable
repository inventory 和 LFS SHA256 为准，不把 README 的泛化描述当作文件名。

## 6. 环境、GPU 与磁盘

### 6.1 当前本地 Codex/Windows

| 项目 | 检测值 |
|---|---|
| OS | Windows NT 10.0.26200.0 |
| 系统 Python | 3.13.5 |
| PyTorch | 未安装 |
| CUDA toolkit / `nvcc` | 未检测到 |
| GPU | 1× NVIDIA GeForce MX450 |
| 显存 | 2,048 MiB total、1,920 MiB free（审计时） |
| Driver | 527.99 |
| C: 可用空间 | 约 275.2 GiB（最新复核） |

本地 MX450 不用于本项目模型计算。系统 Python 缺少 `pytest` 和 NumPy；Phase 0
没有为通过测试而安装依赖。使用 `unittest` 的轻量审计共发现 203 项，其中 190
项通过，13 项仅在 collection/import 阶段因本机缺少 NumPy 失败。该结果不是
远程 `oea-repro` 环境失败。

### 6.2 已验证的远程共享环境

`REMOTE_EVIDENCE`/`COMMITTED_EVIDENCE`：

| 项目 | 检测值 |
|---|---|
| Conda | `/home/jg525/miniconda3/envs/oea-repro`，Python 3.11 |
| PyTorch | 2.7.1+cu126 |
| TorchVision / TorchAudio | 0.22.1+cu126 / 2.7.1+cu126 |
| Transformers | 4.52.4 |
| PEFT / Accelerate | 0.18.0 / 1.10.1 |
| SoundFile | 0.13.1 |
| GPU 验证 | A100-SXM4-80GB 和 RTX 4090 均有成功运行证据 |
| A100 driver/toolkit | Driver 580.95.05；CUDA toolkit 12.6.85 |
| BF16/NCCL | 单卡验证通过 |
| 共享磁盘 | DATA-13B 启动时约 143 TB 可用 |

服务器 hostname/端口是临时的；实验身份只使用 commit、run ID、GPU UUID 和
环境清单。

## 7. 已完成实验结果

当前仓库包含真实 OEA 复现与扩展结果，不得混写为本项目新方法结果：

- OEA-Qwen3B (+Cl) / Clotho T2A、T2T、A2T、UIQ 和效率结果。
- OEA-4 Clotho A2T：R@1/5/10 =
  `27.2727/52.8230/66.6986`，属于官方 checkpoint 的代码扩展，不是论文表格值。
- OEA-6 在 RTX 4090 和 A100 上的显存/延迟控制实验。
- OEA-Qwen3B (+Cl) / Clotho 论文表 2 T2A 的 close reproduction。
- SQuTR DATA-12 下载和 DATA-13A 归档审计。

`results/tables/reproduction_summary.csv` 的既有状态为 48 个 `close` 和 20 个
`blocked` 观察。这里没有 Whisper+BGE、Cross-Encoder、4-best、后验聚合或动态
门控结果。

## 8. Phase 0 发现的协议风险

1. **OEA audio prefix 冲突**：论文描述音频使用 `passage:`；公开 adapter 和
   SQuTR Omni runner 的 audio-only chat 都不加入文本 prefix。必须在 Phase 1
   选择并锁定协议，不能按结果择优。
2. **checkpoint 文件名冲突**：README 的 `step_40.pt` 与 Nemo immutable
   repository inventory 不一致；采用后者并保留差异。
3. **SQuTR query 路径冲突**：真实 ZIP/README 与官方 runner 拼接不一致；采用
   真实归档路径，并将官方 runner 行为记录为实现缺陷。
4. **SQuTR 原始 Omni 不是 OEA**：隐藏维 embedding 不能冒充 OEA 的 512 维
   LoRA+projection 输出。
5. **SQuTR Whisper 只有 1-best**：本项目必须实现独立、可审计的 4-best 和
   score 提取。
6. **BGE query instruction**：模型卡允许无 instruction，但推荐短查询检索长文档
   时添加 instruction。必须在 FiQA dev 前预注册一个设置，不能看 test 选择。
7. **TTS/speaker/noise 未确定**：FiQA train/dev 语音、说话人隔离、噪声来源和
   切分尚未获得批准。动态 gate 训练在该协议锁定前为 `BLOCKED`。
8. **许可证**：`mteb/fiqa` 的官方 card license 字段为 `unknown`。在再分发或公开
   派生音频前必须补充来源许可审查；本地研究下载也需记录这一不确定性。

## 9. Phase 0 结论

项目基础设施可复用，SQuTR 数据已完成归档和解压/CRC 门禁，但内容门禁因并发锁
问题尚未完成；Nemo base 与 +Cl checkpoint 已有完整下载证据。当前可以安全继续
CPU 代码实现和协议测试，但：

- 不能开始 SQuTR embedding，直到 DATA-13C 内容校验成功并生成 final manifest；
- 不能开始正式 Phase 1 GPU，直到 Nemo 实时只读审计、inference-only model lock
  和 audio-prefix 协议确定；
- 不能开始 Whisper/BGE 实验，直到下载获得用户批准；
- 不能训练 gate，直到 FiQA train/dev TTS、speaker/noise 隔离协议获得单独批准。
