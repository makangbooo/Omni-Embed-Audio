# AudioCaps 2.0 数据审计

最后更新：2026-07-17（Asia/Shanghai）

## 结论

论文所称 AudioCaps v2 可以固定到 AudioCaps 官方仓库在 2025-02-24 发布的 `dataset2.0`，不是私有数据集别名。官方仓库 commit `d004db3ea1b01cf4fd0347dd8d27db90cadc8809` 的 validation/test 与论文评测口径一致，但 train 的公开文件存在可重复确认的计数和 CSV 格式冲突。

- `[PAPER]` 训练集为 91,256 samples。
- `[CODE]` 官方 AudioCaps 2.0 README 同样报告 train 91,256、validation 2,475、test 4,875，总计 98,606。
- `[CODE]` 固定 `train.csv` 的 LF 数减 header 恰好是 91,256，但按 OEA 公共 trainer 使用的 `csv.DictReader(..., newline="")` 只有 91,254 个字段完整的记录。
- `[CODE]` 该文件有 3 个未加引号的 bare-CR caption 尾部，解析后成为 3 个缺字段的孤立行；另有 2 个合法的 quoted CRLF 多行 caption。
- `[CODE]` 公共 OEA loader 会跳过 3 个孤立行并保留其前一条被截断的 caption，因此训练入口实际得到 91,254 条。
- `[INFERRED]` 将 3 个孤立片段接回前一 caption 可以修复文本，但样本数仍为 91,254。
- `[MISSING]` 论文没有发布能产生 91,256 个有效训练样本的 manifest/loader；无法确定论文使用了未发布的修正版、额外 2 个样本，还是沿用了物理 LF 计数。

本项目同时生成两个明确分开的训练 manifest：

1. `train_public_loader`：严格遵循公开 OEA loader，91,254 条，保留 3 条截断 caption；
2. `train_repaired_bare_cr`：仅将 3 个孤立尾部接回，仍为 91,254 条，标记 `[INFERRED]`。

在训练开始前必须把二者作为正式 protocol 选择记录下来；无论选择哪个，都不能声称已精确得到论文的 91,256 条。

## 固定来源

| 文件 | bytes | SHA256 |
|---|---:|---|
| `README.md` | 753 | `ae2fcbcf7e4f93964dfaf02c094998b2cfc199a16f4eb0350f65f170ad01a906` |
| `train.csv` | 6,311,901 | `25659eee0ff887972b6a8a74008dfffe582343930e89db71519d66e9dd014f6e` |
| `val.csv` | 169,508 | `dfdd0f83c70fb818fbb86dcba42754abfb97f7774d811fff91a1c89d84d868c5` |
| `test.csv` | 397,498 | `365c8a8a71c9070a8d5dba5dd44f3a781f906632a49cb82b38bebaaa8f3c5ccb` |

来源：[AudioCaps 官方仓库固定 commit](https://github.com/cdjkim/audiocaps/tree/d004db3ea1b01cf4fd0347dd8d27db90cadc8809/dataset2.0)。DATA-06 同时固定 MD5、SHA256 和 byte size，任一不匹配都会失败。

## 解析结果

| split | 官方报告 caption rows | 公共 parser rows | 有效 rows | 唯一音频 | 每音频 captions |
|---|---:|---:|---:|---:|---|
| train | 91,256 | 91,257 | 91,254 | 91,254 | 1 |
| validation | 2,475 | 2,475 | 2,475 | 495 | 5 |
| test | 4,875 | 4,875 | 4,875 | 975 | 5 |

train 的三条孤立片段按固定文件顺序为：

1. `a dog is whimpering`，接在 `audiocap_id=7994` 后；
2. `bang`，接在 `audiocap_id=48170` 后；
3. `speech in the background`，接在 `audiocap_id=39899` 后。

这三条不是独立样本，因为只有 caption 片段，没有 `youtube_id`、`start_time` 或完整四列。

## Evaluation 与 UIQ 对齐

DATA-07 对固定 test CSV 和仓库 UIQ 做了真实全量闭环：

- 四类正向 UIQ 各 975 行、975 个唯一 ID，与 test 的 `${youtube_id}_${start_time}` 集合完全相等；
- test 中 97 个音频存在大小写不敏感的重复 caption（96 个完全相同，1 个只差首字母大小写）；
- 正向 UIQ 的 `original_captions` 精确对应 test captions 的顺序保持、case-insensitive 去重结果；
- negative UIQ 为 630 行、255 个唯一 target ID，全部属于 test；其 `original_captions` 与未去重 test captions 精确相等；
- negative 文件仍没有 hard-negative audio ID，不能直接计算 HNSR/TFR/Δ-Rank。

因此，AudioCaps test 的 metadata/UIQ 候选口径已可审计地固定；尚缺实际音频文件和 negative target/HN pairing。

## DATA-06/07

- `bash scripts/download_data06_audiocaps_v2_metadata.sh`：CPU-only，下载约 6.9 MB 官方 metadata，校验 MD5/SHA256/bytes。
- `bash scripts/run_data07_audiocaps_v2_metadata_validation.sh`：CPU-only，生成四个 metadata-only manifest、统计报告和 SHA256。

输出 manifests：

- `audiocaps_v2_train_public_loader_manifest.jsonl`；
- `audiocaps_v2_train_repaired_bare_cr_manifest.jsonl`；
- `audiocaps_v2_validation_manifest.jsonl`；
- `audiocaps_v2_test_manifest.jsonl`。

## 仍需资源

- `[CODE]` 官方 README 要求填写 [AudioCaps 原始音视频申请表](https://forms.gle/2wF54Y1Ft2LtPdhW8)；Git 仓库只发布 metadata。
- 若官方未提供打包音频，需要按固定 `youtube_id/start_time` 下载并逐条记录可用性。YouTube 下架会导致当前可获取候选集小于论文候选集，必须单独报告，不能静默缩小测试集。
- 训练开始前需决定采用公共 loader 口径还是 bare-CR 修复口径；这只是公开资源复现选择，不能解决缺失的两条计数差异。
