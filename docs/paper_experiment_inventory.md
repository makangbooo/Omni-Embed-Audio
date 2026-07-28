# Omni-Embed-Audio 论文实验清单

## 审计口径

- 论文：*Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval*，ACL 2026，17 页（正文、参考文献、附录 A–M）。
- 论文 PDF SHA256：`E76BBD96C82AC78568ED75EAEDCBC62E389358C4BB226457AC45B55AB63CFE38`。
- 官方源码快照：`JudeJiwoo/Omni-Embed-Audio@b261ad0743dbfac67cb6b20016fd951a366b3f35`（2026-05-18）。
- 计数规则：25 个需要独立执行/采集的实验协议，3 个派生汇总分析，3 个图或方法核验项，共 31 个可审计条目。表 5 与表 16 是同一效率实验的正文/附录版本，不重复计为两组。
- 来源标签：`[PAPER]` 论文明确给出；`[CODE]` 官方代码、配置、官方模型仓库或发布数据给出；`[INFERRED]` 为待验证推断；`[MISSING]` 论文和代码均未给出。
- 可复现性：A=官方代码和数据完整；B=需要额外公开资源或小型兼容补丁；C=缺少配置/映射，只能近似或需作者补充；D=人工评测或闭源 API，不能完全自动复现。

截至 2026-07-28 的提交内证据重校准计数为：`COMPLETED=4`、`IN_PROGRESS=10`、`TODO=5`、`BLOCKED=12`，总计 31。该计数不包含 SpeechXBT-OEA、FiQA、NQ、SQuTR、ASR reranker 或 A2T 扩展。910 个论文指标聚合后的模型×数据集×任务矩阵见 `results/tables/paper_experiment_matrix.csv`；其中 `PAPER` 与复现状态分列，避免把论文转录值误写成复现值。

## 全量清单

| ID | 论文位置 | 实验 | 数据集 | 模型 | 指标 | 官方命令 | 所需资源 | 可复现性 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| FIG-01 | 图 1 | 代表模型综合结果可视化 | AudioCaps、Clotho、MECAT | OEA-Qwen7B (+Cl)、M2D-CLAP | 三数据集 mean R@5、HNSR@10 | `[MISSING]` 无绘图脚本 | DER-01、DER-02、EXP-17 的结果 | B（派生图） | TODO |
| FIG-02 | 图 2；§3.1；附录 A | 架构核验：共享 backbone、LoRA、双投影头、512 维、L2 | 5 个附带 Clotho 样例可做 smoke | 三种 OEA backbone | 结构与参数形状 | `[CODE]` 锁绑定生成器严格加载 LoRA/双 head；`[OBSERVED]` Qwen3B 与 Nemo3B 独立 5 音频/25 文本 smoke 均 exit 0、输出 512 维 | `results/audits/fig02_architecture_verification_20260728.json`；Qwen7B runtime 仍 `[MISSING]`，不计为 Qwen7B 评测完成 | B（方法/张量契约已核验） | COMPLETED |
| FIG-03 | 图 3；§3.3；附录 L | 否定查询指标示意与人工构造单元测试 | 合成排名样例 | 与模型无关 | R@k、HNSR、HNSR@k、TFR、TFR-HN@k、Δ-Rank | `[CODE]` `AudioRetrieval/evaluation/negative_metrics.py`、`negative_canonical.py`；`python -m unittest tests.test_negative_metrics tests.test_evaluate_negative_embedding_artifacts -v` | 无 GPU；测试向量 | A（公式、显式配对评测与合成验证完整） | COMPLETED |
| EXP-01 | §3.2.2；附录 D；表 8–9 | 生成五类 UIQ | AudioCaps、Clotho、MECAT captions + HN captions | GPT-5.1 | 输出数量、格式、长度约束、语义有效率 | `[CODE] python -m AudioRetrieval generate-uiq ...`，但默认 GPT-4/0.7/100 tokens 且 prompt 不同 | GPT-5.1 API、完整 captions、HN 配对 | D（闭源 API；官方代码不匹配论文 prompt） | TODO |
| EXP-02 | §3.2.3；附录 C/E；表 1、7 | 人工 UIQ 有效性评测 | 75 样例 × 5 类型；9 人；675 ratings | 人工标注者 | 5 点 Likert 均值/标准差、分数据集结果 | `[MISSING]` 无界面、样例抽样清单或原始 ratings | 75 样例 ID、音频、标注者与协议 | D | BLOCKED |
| EXP-03 | §3.2.3；附录 E；表 1、10 | LLM UIQ 有效性评测 | 与 EXP-02 相同的 75 样例 | Claude Opus 4.5 | 5 点 Likert、Human–LLM agreement（正文报告 r/p） | `[MISSING]` 无评测脚本或原始响应 | Claude Opus 4.5 API、75 样例清单 | D | BLOCKED |
| EXP-04 | §3.2.3；附录 I | UIQ 与真实 Freesound 查询 token-length 分析 | UIQ 13,053 条；Freesound 查询统计 | 分词器 `[MISSING]` | token 数分布、均值；对照文献 1.8 tokens | `[MISSING]` 无脚本、无真实查询日志/分词定义 | Freesound 查询数据或文献可复算统计 | C | BLOCKED |
| EXP-05 | §3.3；附录 K | 四阶段 hard-negative mining + 人工复核 | 三个评测集 | MGA-CLAP + BGE-large-en-v1.5 | Top-20、声学相似度、语义相似度、保留率、最终配对数 | `[CODE] preprocess hard-negatives` 默认 Top-50；另有 LAION-CLAP 替代脚本 Top-20 | MGA-CLAP 权重、BGE、完整音频/captions、人工复核记录 | C（动态阈值与最终配对未发布） | BLOCKED |
| EXP-06 | 附录 B.1；表 6 | 数据来源与潜在污染关系审计 | WavCaps 子集与 7 个评测集 | 文件/来源匹配 | 来源对应关系 | `[CODE]` DATA-09 已实现 AudioCaps/Clotho metadata provenance；MECAT 仍待 DATA-04/05 | 固定 WavCaps/AudioCaps/Clotho metadata；MECAT manifest | B | IN_PROGRESS |
| EXP-07 | §4.1；附录 B.2 | AudioCaps test–WavCaps AudioSet_SL 重叠 | 975 AudioCaps test clips；108,317 WavCaps AudioSet_SL | 规范化 YouTube ID 精确匹配 | 173/975=17.7%；865 caption rows；占 WavCaps 0.16% | `[CODE]` DATA-09 在固定 revision 全量元数据上实现并复算 | 固定 WavCaps AudioSet_SL metadata | B | COMPLETED |
| EXP-08 | §4.1；附录 B.3 | Clotho evaluation–WavCaps Freesound 重叠 | 1,045 Clotho evaluation clips | filename case-insensitive 精确匹配；可扩展 fingerprint | 638/1,045=61.0% | `[CODE]` DATA-09 精确复现 638 个匹配文件名；这些文件名映射到 1,017 个 WavCaps 候选，64 个文件名有歧义 | Clotho v2.1 metadata、WavCaps FreeSound metadata | B | COMPLETED |
| EXP-09 | §4.1；附录 B.4 | MECAT–WavCaps 无显著重叠核验 | MECAT 与 WavCaps | filename + embedding 检查，具体阈值 `[MISSING]` | 重叠数/率 | `[CODE]` `bash scripts/run_data11_mecat_wavcaps_provenance.sh` 已实现严格 source-video ID 候选审计；本地发布 UIQ 对固定 metadata 得 4 个同源视频，但非音频重复结论 | DATA-05 canonical 848 manifest、论文 847 排除 ID `[MISSING]`、完整音频、论文嵌入模型/阈值 `[MISSING]` | C | IN_PROGRESS |
| EXP-10 | §5.1；表 2 | caption Text-to-Audio 完整基线 | AudioCaps 975、Clotho 1,045、MECAT 847（发布 UIQ 为 848，需澄清） | 4 CLAP + 3 vanilla LALM + 6 OEA | R@1/5/10 | Qwen3B-Cl/Clotho 与 Qwen3B-AC/Clotho 均完成全量 embedding 和同构四协议闭环；Qwen3B-AC `[CODE]` all-caption=`19.0048/41.9522/55.9617`、seed0=`19.3301/41.7225/55.6938`，论文=`19.18/42.05/55.85` | 其余 11 个模型/权重、AudioCaps/MECAT 候选集、论文 caption 选择 `[MISSING]` | C（两个 OEA 分栏闭环完成） | IN_PROGRESS |
| EXP-11 | §5.2；表 3 | caption Text-to-Text 完整基线 | 同 EXP-10 captions/candidates | 同 EXP-10 | R@1/5/10 | Qwen3B-AC seed0 `[CODE]`=`61.3397/74.1627/79.3301`，all-caption `[INFERRED]`=`62.7943/73.8756/78.3158`，论文=`62.81/73.76/78.22`；与 Qwen3B-Cl 一样，全部协议均预声明并分栏，未按接近度择优；Nemo3B-Cl 的同构四协议配置已固定，等待远程 embedding 可访问后 CPU finalization | 其余模型/数据；论文 caption 选择、self-exclusion、tie 口径 `[MISSING]` | C（两个 OEA 分栏闭环完成；一个 OEA 为 PARTIAL） | IN_PROGRESS |
| EXP-12 | 附录 G.1；表 12 | Question UIQ T2A | AudioCaps/Clotho/MECAT-UIQ | 4 CLAP + 6 OEA | 每数据集 R@1/5/10 | 上游 README/CLI 不兼容；复现分支已提供锁绑定 GPU 生成和 CPU 四协议套件；Nemo3B-Cl 的模型锁、四类 query hash、进度/ETA 与 tmux 保留协议已固定但尚未运行 | Qwen3B-Cl/Clotho=`25.8373/54.0670/66.6029`（论文 `25.74/54.26/66.79`）；Qwen3B-AC/Clotho=`21.5311/44.4976/59.5215`（论文 `21.34/44.11/59.71`）；其余模型/数据待准备 | B（数据公开，需兼容补丁） | IN_PROGRESS |
| EXP-13 | 附录 G.2；表 13 | Imperative UIQ T2A | 同 EXP-12 | 同 EXP-12 | 每数据集 R@1/5/10 | 同一固定生成/评测套件显式物化 Imperative 的 1,045 个 query indices | Qwen3B-Cl/Clotho=`25.8373/55.6938/66.9856`（论文 `25.45/55.69/67.56`）；Qwen3B-AC/Clotho=`22.4880/46.2201/59.1388`（论文 `22.30/46.12/59.33`）；其余同 EXP-12 | B | IN_PROGRESS |
| EXP-14 | 附录 G.3；表 14 | Paraphrase UIQ T2A | 同 EXP-12 | 同 EXP-12 | 每数据集 R@1/5/10 | 同一固定生成/评测套件显式物化 Paraphrase 的 1,045 个 query indices | Qwen3B-Cl/Clotho=`26.9856/55.1196/70.2392`（论文 `27.56/55.50/70.81`）；Qwen3B-AC/Clotho=`20.9569/46.7943/60.0957`（论文 `21.24/46.79/60.19`）；其余同 EXP-12 | B | IN_PROGRESS |
| EXP-15 | 附录 G.4；表 15 | Keyphrase/`tagging` UIQ T2A | 同 EXP-12 | 同 EXP-12 | 每数据集 R@1/5/10 | 同一套件保留 `[CODE] tagging` 与 `[PAPER] Keyphrase` 双标签并显式物化 1,045 个 query indices | Qwen3B-Cl/Clotho=`27.5598/58.0861/71.6746`（论文 `27.66/57.99/71.87`）；Qwen3B-AC/Clotho=`24.3062/49.6651/61.7225`（论文 `24.50/49.76/61.72`）；其余同 EXP-12 | B | IN_PROGRESS |
| EXP-16 | §5.3.2；表 17 | Negative UIQ 普通检索 | 630/542/409 条 negative rows | 4 CLAP + 6 OEA | mean R@5、R@10 | `NegativeQueryRunner` 读取旧 schema，且不是论文指标实现 | 音频、negative queries、target 映射 | C | BLOCKED |
| EXP-17 | §3.3/§5.3.2；附录 L/M；表 4、17 | Hard-negative discrimination | 同 EXP-16 + target/HN 音频 ID | 4 CLAP + 6 OEA | Δ-Rank、HNSR、HNSR@10、TFR、TFR-HN@10 | `[CODE]` 复现分支提供 `scripts/evaluate_negative_embedding_artifacts.py` 和 CPU wrapper，强制显式完整配对并保存全量排名/哈希；上游源码仍无这些指标，发布 JSONL 仍无 HN audio ID | 作者 target–HN 配对或另行标记的可审计重建映射 | C | BLOCKED |
| EXP-18 | §5.4；附录 H；表 5、16 | 推理效率 | Clotho 1,045 clips | LAION/MGA/M2D + 3 OEA | audio ms/clip、text ms/query、peak GPU GB | `[OBSERVED]` Qwen3B(+Cl) 已完成 A100 与 RTX 4090 受控 benchmark；`[MISSING]` 论文 batch/warmup/timing/memory 口径，不能提升为严格表 5/16 复现 | Qwen3B(+Cl) 受控结果及其余 5 模型；固定论文协议仍 `[MISSING]` | C（一个模型 CONTROLLED_ONLY） | IN_PROGRESS |
| EXP-19 | §3.1/§5.4；表 5、16 | 训练参数量/占比 | 不依赖数据 | 3 OEA + 3 CLAP | trainable params、占总参数百分比 | `[CHECKPOINT][OBSERVED]` Qwen3B(+Cl) LoRA=12.61568M、双 head 各 1.04960M、合计 14.71488M；`[MISSING]` 论文 16.2M 统计口径 | 其余模型结构/权重；Qwen3B(+Cl) 只作为 checkpoint 实测分栏 | B（一个模型 CONTROLLED_ONLY） | IN_PROGRESS |
| EXP-20 | §4.1/4.2；附录 A | OEA-Nemo3B：WavCaps → AudioCaps | WavCaps 275,618（≤31s、去泄漏）→ AudioCaps v2 91,256 | Omni-Embed-Nemotron-3B | train loss、val R@10、表 2/3/12–17 | `[MISSING]` 无 Nemo 完整 stage config；通用 trainer 可近似 | 数据、base 权重、blocklists、完整超参 | C | BLOCKED |
| EXP-21 | §4.1/4.2；附录 A | OEA-Nemo3B (+Cl)：追加 Clotho v2 | EXP-20 + 3,839 Clotho clips | 同上 | 同上 | `[CODE]` `python -m AudioRetrieval.training.oea.train_omniembed_lora --dataset clotho --train-csv ...development... --val-csv ...evaluation...`；论文未命名 early-stopping split | EXP-20 checkpoint、DATA-10 manifests、论文 split 证据 `[MISSING]` | C | BLOCKED |
| EXP-22 | §4.1/4.2；附录 A | OEA-Qwen3B：WavCaps → AudioCaps | 同 EXP-20 | Qwen2.5-Omni-3B | 同 EXP-20 | `[CODE] wavcaps_3b.sh` 仅覆盖 WavCaps，未给 AudioCaps stage | 数据、权重、完整超参、DDP 修复 | C | BLOCKED |
| EXP-23 | §4.1/4.2；附录 A | OEA-Qwen3B (+Cl)：追加 Clotho v2 | 同 EXP-21 | Qwen2.5-Omni-3B | 同上 | `[CODE]` 同 EXP-21 的通用 Clotho launcher；backbone/stage 超参未固定 | EXP-22 checkpoint、DATA-10 manifests、完整 Qwen stage config `[MISSING]` | C | BLOCKED |
| EXP-24 | §4.1/4.2；附录 A | OEA-Qwen7B：WavCaps → AudioCaps | 同 EXP-20 | Qwen2.5-Omni-7B | 同 EXP-20 | `[MISSING]` 无 7B stage config | 数据、权重、完整超参、8 卡训练 | C | BLOCKED |
| EXP-25 | §4.1/4.2；附录 A | OEA-Qwen7B (+Cl)：追加 Clotho v2 | 同 EXP-21 | Qwen2.5-Omni-7B | 同上 | `[CODE]` 同 EXP-21 的通用 Clotho launcher；7B stage 超参未固定 | EXP-24 checkpoint、DATA-10 manifests、完整 7B stage config `[MISSING]` | C | BLOCKED |
| DER-01 | 附录 F；表 11 | 三数据集 T2A/T2T 均值汇总 | EXP-10、EXP-11 | 所有基线/OEA | mean R@1/5/10 | `[MISSING]` 无汇总脚本 | EXP-10/11 raw metrics | B（派生） | TODO |
| DER-02 | §5.3；表 4 | 三数据集 UIQ 均值与 Avg UIQ | EXP-12–17 | 4 CLAP + 6 OEA | 四类 R@5、negative HNSR@10、Avg UIQ | `[MISSING]` 无表生成脚本 | UIQ 分数据集 raw metrics | B（派生） | TODO |
| DER-03 | §5.4；附录 J | backbone 泛化与 retrieval-specific scaling | 表 2/3/11、EXP-18/19 | Nemo3B、Qwen3B、Qwen7B | 跨数据集趋势、3B/7B 差值、效率 | `[MISSING]` 无分析脚本 | 已复现主表及效率结果 | B（派生） | TODO |

## 官方仓库覆盖结论

### 已提供

- `[CODE]` 六个 Hugging Face 模型仓库和 base-model 映射；根 README 所称“每个仓库包含 `step_40.pt`”不符合实际文件名。复现分支已按不可变 revision/LFS metadata 固定实际的 `step_400_best.pt`、`step_450_best.pt`、`step_350.pt`、`step_40.pt`、`step_300.pt`、`step_330.pt`，并由 `configs/checkpoints/official_oea_checkpoints.json` 统一解析。
- `[CODE]` 13,053 条 UIQ JSONL：Question、Imperative、Paraphrase、`tagging`、Negative，覆盖三个数据集。
- `[CODE]` OEA adapter、LoRA attachment、双投影头、mean pooling、L2 normalization、InfoNCE trainer、WavCaps manifest builder。
- `[CODE]` LAION-CLAP、Robust-CLAP、MGA-CLAP、M2D-CLAP adapter 文件；但统一 CLI/Hydra 未暴露全部 adapter，权重获取与 revision 未固定。
- `[CODE]` 两套 hard-negative mining 路径：MGA-CLAP 通用 pipeline（默认 Top-50）与 LAION-CLAP 预计算替代脚本（Top-20）。

### 关键缺口

1. `[CODE]` README 的 OEA 评测命令把 Hydra `key=value` 参数传给 argparse CLI，按当前入口会直接报错；若改为直接执行 `eval_hydra.py`，仍只加载裸 backbone，未加载 LoRA 和 projection heads，不能生成表 2–4/12–17 的 OEA 数字。
2. `[CODE]` `python -m AudioRetrieval train` 只打印提示并返回成功，实际训练必须直接调用 trainer。
3. `[CODE]` 统一 evaluator 不支持 MECAT；官方仓库也没有 MECAT dataset config/manifest。
4. `[CODE]` 发布 UIQ schema 是 `audio_id/generated_query/query_type`；`UIQRunner` 读取旧的 `clip_id/uiq[].bucket/query` schema。
5. `[CODE]` 三个 negative JSONL 的 1,581 行均没有 hard-negative audio ID，只有 `negative_captions`；无法直接计算 HNSR/TFR/Δ-Rank。
6. `[CODE]` 上游 README 声称 `evaluation/` 实现 HNSR/TFR/Δ-Rank，但上游 Python 源码没有这些函数；复现分支已在 `88a7d1f`/`cd4d00b` 补齐公式、显式配对 evaluator、失败保护和审计工件，这不补回缺失的官方 HN audio ID。
7. `[PAPER]` 训练使用 PyTorch DDP、BF16、validation R@10 early stopping；`[CODE]` trainer 无 DDP、无 seed、无 scheduler/warmup/grad clipping，autocast 未显式指定 BF16，并按 validation loss 早停。
8. `[PAPER]` 音频输入使用 `passage:`；`[CODE]` `_build_audio_messages` 明确忽略 `passage_prefix`。这是论文方法与公开实现的实质差异，需用官方 checkpoint smoke test 判定实际训练口径。
9. `[PAPER]` WavCaps 写作过滤到 `<=31` 秒并报告 275,618 条；固定公开元数据按该条件为 275,691，只有 `[INFERRED]` `0 < duration < 31` 精确得到 275,618。DATA-09 已精确复现 173 个 AudioCaps 和 638 个 Clotho 重叠，但论文 blocklist、Clotho 重复文件名消歧规则和最终训练 manifest 仍 `[MISSING]`。
10. `[PAPER]` hard-negative Stage 2 使用动态声学阈值保留约 3×最终数量；`[CODE]` 替代脚本直接保留 Top-3，通用 pipeline 默认 Top-50，均不能证明等同论文。
11. `[PAPER][CODE]` AudioCaps v2 官方 README 与论文均写 91,256 train；DATA-06/07 已固定官方 commit，但公共 OEA `csv.DictReader` 只能产生 91,254 个有效记录。3 个 bare-CR caption 尾部可修复但计数仍为 91,254；论文有效 91,256-row manifest `[MISSING]`。
12. `[PAPER]/[CODE]` 正文 MECAT 为 847 对，UIQ 发布为 848 个正查询 ID；需作者 manifest 解释 1 条差异。
13. `[PAPER]` `+Cl` 使用 3,839 clips 并按 validation R@10 早停，但没有命名 Clotho split；`[CODE]` 唯一 launcher 明确用 development 训练、evaluation 早停，而不是 official validation。DATA-10 会完整保留两个官方训练侧 split；公开代码路线与无污染 official-validation 敏感性路线必须分开报告。

## 第一轮结论

- A 类：没有一张论文主表可在当前公开仓库中“不补代码、不补映射”直接完整复现。
- B 类：架构 smoke、正向 UIQ 四类型、数据来源/精确重叠、参数量与派生表；需要公开数据和小型兼容补丁。
- C 类：论文主表评测、negative discrimination、三阶段训练、效率、MECAT、hard-negative mining；需要补实现且部分配置/映射缺失。
- D 类：GPT-5.1 UIQ 再生成、Claude Opus 4.5 评测、9 人人工评测。
