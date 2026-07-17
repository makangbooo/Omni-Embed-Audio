# MECAT 数据审计

最后更新：2026-07-18（Asia/Shanghai）

## 结论

现有公开证据强烈指向 MECAT-Caption 的 `00A`（不含语音/音乐的一般声音）测试域，但在 DATA-05 完成 archive/UIQ ID 精确集合匹配前仍标为 `[INFERRED]`；公开证据只能确定 848 条候选，不能确定论文表 2/3 中的精确 847 条子集。

- `[PAPER]` §4.1 与表 2/3 将 MECAT 评测集写为 847 个自动 caption 的音频–文本对。
- `[CODE]` 官方 MECAT-Caption 数据卡将 `00A/test` 记为 848 条。
- `[CODE]` OEA 发布的 MECAT question、imperative、paraphrase、tagging 四个 JSONL 各有 848 个唯一 `audio_id`，每行 metadata 均声明 `domain=00A`。
- `[MISSING]` 论文和 OEA 仓库都没有公布被排除的 1 个 ID、过滤条件或 847-row manifest。
- `[MISSING]` MECAT JSON 同时提供 `long`、`short`、`speech`、`music`、`sound`、`environment` 六组字段，论文和公开代码没有说明表 2/3 的 T2A/T2T 使用哪个字段或怎样组合字段。

因此，本项目不会任意删除样本或任选 caption 字段来“对齐”论文。DATA-05 会先保留全部 848 条和全部六个字段，完成音频、metadata 与 UIQ ID 的精确审计；MECAT 的严格 847 条 T2A/T2T 仍标为 `BLOCKED`。若后续只能做公开资源实验，会把 848 条结果明确标记为替代口径。

## 固定资源

| 项目 | 固定值 | 来源 |
|---|---|---|
| 数据仓库 | `mispeech/MECAT-Caption` | [官方 Hugging Face 数据集](https://huggingface.co/datasets/mispeech/MECAT-Caption) |
| revision | `be4a24c3f7309d74208e08a7cce49e72cb7a5834` | `[CODE]` 官方数据仓库不可变 commit |
| 配置/split | `00A/test` | `[CODE][INFERRED]` 数据卡数量与 OEA UIQ 的 domain/count 对应；下载后必须再做 ID 集合精确验证 |
| 文件 | `00A/test_0000-0000000.tar.gz` | `[CODE]` 官方 WebDataset shard |
| 大小 | 173,168,424 bytes | `[CODE]` 固定 revision 的 Hugging Face 文件 metadata |
| LFS SHA256 | `644cf75e2509c633452a18e36c41b285a317c6cbc06198d7dfe406c5aa5122c4` | `[CODE]` 固定 revision 的 Hugging Face LFS metadata |
| 官方实现仓库 | commit `a004949d58e86e2ee56baa879607ec2109cfcc46` | [xiaomi-research/mecat](https://github.com/xiaomi-research/mecat) |

只下载 `README.md` 和一个 `00A/test` shard，避免无目的下载约 16 GB 的全部 MECAT 域。

## DATA-04：下载与来源校验

入口：`bash scripts/download_data04_mecat_00a_test.sh`

该步骤：

1. 通过 dataset API 而非 model API 验证 repo 类型；
2. 要求远端解析出的 commit 与固定 revision 完全相同；
3. 下载前核对 shard 的大小和 LFS SHA256 metadata；
4. 使用单 worker、最多 20 次退避重试，保留 `.incomplete` 文件以便断点续传；
5. 下载后重新计算本地 SHA256；
6. 将命令、Git commit、主机、下载尝试、文件清单和校验结果写入独立日志目录。

DATA-04 不解压、不使用 GPU，也不覆盖已有无 revision marker 的非空目录。

## DATA-05：安全解压与完整性校验

入口：`bash scripts/run_data05_mecat_validation.sh`

该步骤：

1. 解压前再次核对固定 SHA256；
2. 拒绝绝对路径、`..`、反斜杠、符号链接、硬链接和其他非普通文件；
3. 要求恰好 848 个 sample ID，且每个 ID 恰好有一个 FLAC 和一个 JSON；
4. 解压到隔离 staging 目录，成功后才原子移动为正式目录；
5. 逐个完整解码 FLAC，检查 frame 数和有限值；
6. 要求每个 JSON 包含六个官方 caption 字段，检测重复 JSON key，但不擅自选择检索 caption；
7. 对四个正向 UIQ 文件执行 848-ID 精确集合匹配；
8. 对 409 条 negative UIQ 检查 audio ID 均属于候选集，但不伪造缺失的 hard-negative audio ID；
9. 生成 848-row JSONL manifest、统计 JSON、逐文件 SHA256 和完整运行日志。

## DATA-11：MECAT–WavCaps 来源视频审计

入口：`bash scripts/run_data11_mecat_wavcaps_provenance.sh`

论文附录 B.4 的音频/embedding 重叠实现、模型、阈值、人工复核候选均未公开。DATA-11 先完成无需猜测的来源级子问题：

1. 严格从 MECAT `sample_id` 的前 11 个字符解析公开的 YouTube source-video ID，但不猜测后续时间编码；
2. 将 WavCaps AudioSet_SL 的 `Y<11-char>.wav` 规范化为同一 ID；
3. 只做精确 ID 交集，不使用 caption 相似度或模糊字符串匹配；
4. 报告同一来源视频的全部 MECAT segments，但不据此删除样本或修改训练/评测集合；
5. canonical JSONL 若已存在且内容不同则拒绝覆盖。

在 DATA-05 远程 archive 校验前，先用发布的 848 条 MECAT question UIQ ID 和固定 WavCaps metadata 完成本地全量审计：

| 项目 | 结果 | 证据等级 |
|---|---:|---|
| MECAT UIQ sample IDs | 848 | `[CODE]` |
| MECAT 唯一 source videos | 807 | `[CODE][INFERRED]` ID 结构解析 |
| 具有多个 MECAT segments 的 source videos | 37 | `[CODE][INFERRED]` |
| 与 WavCaps AudioSet_SL 同源的 YouTube IDs | 4 | `[CODE][INFERRED]` |
| 涉及 MECAT samples | 4 | `[CODE][INFERRED]` |

4 个候选 source-video IDs 为 `FQIZHO6l0IY`、`Nw2EarZypA0`、`qEfTLLEpojc`、`vzt3AXNeKIQ`。它们只证明同一来源视频，不能证明 MECAT 与 WavCaps 使用了相同时间段或相同音频字节，也不能替代论文的 embedding 检查。小型证据保存在 `results/data_audits/data11_mecat_wavcaps_provenance_local.json`；DATA-05/08 完成后由 DATA-11 远程复算并确认 archive manifest。

## 尚未解决的问题

| 问题 | 标记 | 对实验的影响 | 解决条件 |
|---|---|---|---|
| 论文 847 与公开 848 的差异 | `[MISSING]` | 不能宣称精确复现 MECAT 表 2/3/12–15 | 作者提供 847-row manifest、排除 ID 或可审计过滤规则 |
| T2A/T2T 的 caption 字段/组合 | `[MISSING]` | 不能生成论文同口径的 MECAT caption query/candidate | 作者提供字段选择或原始评测 manifest |
| negative 的 hard-negative audio ID | `[MISSING]` | 不能精确计算 MECAT HNSR/TFR/Δ-Rank | 作者提供 target/HN pairing，或另做明确标为 `[INFERRED]` 的重建实验 |
| 附录 B.4 的音频/embedding 重叠协议 | `[MISSING]` | 4 个同源视频候选不能判定为音频重复 | 作者提供模型、阈值、时间片/候选清单与人工复核结果；或在完整音频上做明确标为替代协议的实验 |
