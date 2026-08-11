# Omni-Embed-Audio 复现状态

最后更新：2026-08-11（Asia/Shanghai）

## 范围声明

当前代码树只维护 OEA 论文复现。官方论文之外的后续方法探索、专用数据、
训练 checkpoint、日志和执行入口均已移除，不计入状态、待办或完成度。
主实验使用作者发布的官方 checkpoint 推理；OEA 从头训练尚未复现。

## 总体结论

- 论文 31 行 inventory 的最后一次严格整行快照：`4 COMPLETED / 13 IN_PROGRESS / 4 TODO / 10 BLOCKED`，即 `4/31 = 12.9%`。
- 该整行比例不能代表 public-controlled cell 覆盖。Clotho、AudioCaps、MECAT public-848 的主要受控矩阵已经执行。
- 正向 UIQ 和 Negative UIQ 的三数据集十模型受控矩阵已经完成。
- 缺少论文 caption selection、MECAT 847-row manifest、Negative UIQ exact 音频配对和效率计时细节，因此相关 strict claim 保持 blocked。

## 远程执行汇报格式

任何需要用户执行命令、提供资源或返回远程结果的回复，必须依次包含：

1. `【需要你执行的操作】`
2. `【执行后请告诉我】`
3. `【实验结果与论文对比】`
4. `【实验进度】`
5. `【我目前正在做什么】`

命令前必须在命令前以独立字段显式列出：

- `是否使用 GPU：是/否`
- `是否使用 OEA 论文的源代码文件：是/否`
- `预计执行时间：<可审计的时间范围>`

不得只把 GPU、OEA 官方源码使用情况或预计时间埋在说明段落中。实验对比必须
列出论文值、复现值、绝对差、协议来源和判定；有多个预声明协议时不得按接近论文数值择优。
非论文表格实验也必须单独列出结果和证据边界。没有新结果时
明确写“本轮无新实验结果”。

## 当前状态表

| 阶段 | 内容 | 状态 | 证据边界 |
|---|---|---|---|
| 项目审计 | 论文、附录、官方代码、31 行 inventory、约 910 个纸面指标 | COMPLETED | 论文值和复现值分栏 |
| 环境 | Conda lock、CPU/GPU、CUDA/BF16、音频 I/O | COMPLETED | A100 与 RTX 4090 均有运行记录 |
| 官方模型 | Nemo3B/Qwen3B/Qwen7B 的 AC/Cl 六变体 | COMPLETED | revision、checkpoint、model lock 与 loadability 已审计 |
| Figure 2 | LoRA、双 projection head、512-d 张量契约 | COMPLETED | `fig02_architecture_verification_20260728.json` |
| Clotho 数据 | 1,045 WAV、5,225 captions、UIQ 对齐 | COMPLETED | v2.1 公开修复版；论文未给归档 checksum |
| AudioCaps 数据 | test 音频与 captions | COMPLETED/CONTROLLED | train 91,254 vs 论文 91,256 |
| MECAT 数据 | public 848 音频、captions、UIQ | COMPLETED/CONTROLLED | 论文 847-row 排除 ID 未公开 |
| WavCaps 数据 | metadata、时长、来源、leakage 候选 | COMPLETED | 严格论文训练过滤仍有缺失信息 |
| Table 2/3 主检索 | 三数据集、OEA/CLAP/vanilla 受控矩阵 | COMPLETED/CONTROLLED | caption selection 与 T2T 细节未公开 |
| Tables 12–15 正向 UIQ | 三数据集十模型矩阵 | COMPLETED/CONTROLLED | 公开 UIQ 与公开音频协议 |
| Tables 4/17 Negative UIQ | 三数据集十模型矩阵 | COMPLETED/CONTROLLED | 1,581/1,581 deterministic pairing；exact target/HN 配对 blocked |
| Tables 5/16 效率 | RTX 4090 受控测量和 A100 边界 | COMPLETED/CONTROLLED | 论文 timing/memory/counting 口径不完整 |
| 官方 OEA 训练 | 按论文配置从头训练 | BLOCKED | 完整训练数据/过滤/细节和训练闭环未复现 |

## 关键模型证据

| 审计 | 状态 | 证据 |
|---|---|---|
| 九个官方 OEA/base 资产 metadata-only presence candidate | COMPLETED | `results/audits/official_oea_model_presence_scan_20260729.json`；只证明 presence，不提升为内容完整性 |
| OEA-Nemo3B-AC 按变体内容 hash/provenance 审计 | COMPLETED | `results/audits/oea_nemo3b_ac_model_resource_audit_20260729.json`；25 个文件、18,889,955,098 bytes 均通过 |
| Figure 2 方法与张量契约 | COMPLETED | `results/audits/fig02_architecture_verification_20260728.json` |

## 代表性复现值

| 实验 | 复现值 R@1/5/10 | 判定 |
|---|---|---|
| OEA-Nemo3B-Cl / Clotho T2A all-caption | 21.7225 / 47.1196 / 60.4402 | `CONTROLLED/CLOSE` |
| OEA-Nemo3B-Cl / Clotho T2A seed-0 | 21.6268 / 46.7943 / 59.8086 | `CONTROLLED/CLOSE` |
| OEA-Nemo3B-Cl / Clotho T2T seed-0 | 62.9665 / 75.9809 / 80.8612 | `CONTROLLED/CLOSE` |
| OEA-Qwen7B-Cl / Clotho T2A all-caption | 22.0287 / 48.5359 / 61.6459 | `CONTROLLED` |
| vanilla Nemotron-3B / Clotho T2A all-caption | 7.2536 / 21.3014 / 30.1818 | `CONTROLLED/CLOSE` |

完整结果以 `results/observations/reproduction_observations.jsonl`、
`results/tables/paper_experiment_matrix.csv` 和各 `results/audits/*.json`
为准，不能用本页摘要替代具体协议。

## 必须保留的远程资产

- `/home/jg525/Omni-Embed-Audio`：当前主仓库和复现运行产物。
- `/home/jg525/models/oea` 中的官方 OEA/base/CLAP 模型及模型锁对应文件。
- `/home/jg525/datasets/oea` 中的 Clotho、AudioCaps、MECAT、WavCaps/UIQ 数据。
- 论文主表、UIQ、Negative UIQ、基线和效率评测的 `logs/`、`results/raw/` 与 `outputs/experiments/` 工件。
- `/home/jg525/repro_artifact_backups` 暂留到 model-lock 最终归档完成。

## 剩余严格缺口

1. 作者未公开的 caption selection 和 T2T self/tie 规则。
2. MECAT 论文 847 条的排除 ID 与 caption 构造。
3. Negative UIQ 的 exact target/hard-negative 音频配对。
4. Tables 5/16 的完整 timing、warmup、batch、memory 和参数统计口径。
5. OEA 论文训练数据过滤和从头训练闭环。

这些缺口不能通过挑选更接近论文的结果、猜测配对或补造样本来解除。
