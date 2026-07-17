# Clotho development/validation 数据审计

## 结论

- `[PAPER]` 论文说明 `+Cl` 阶段追加 3,839 个 Clotho v2 clips，并以 validation R@10 早停，但没有说明这里的 “validation” 是 Clotho 官方 validation 还是 evaluation split。
- `[CODE]` 唯一公开的 Clotho 追加训练命令使用 development 训练，却把 evaluation 作为 `--val-csv` 和 `--val-audio-dir`。这会让 checkpoint 选择直接查看后续论文评测使用的 Clotho evaluation 集，属于需要单独披露的评测集早停污染风险。
- `[CODE]` DATA-10 固定修复后的 Clotho v2.1 Zenodo record `4783391`，同时保留 development、validation 两个官方 split。它不会把 validation 静默替换进公开命令，也不会把 evaluation 混入这两个新目录。
- `[MISSING]` 论文真实使用的早停 split、随机种子和逐 epoch checkpoint 选择记录均未发布。因此严格复现与无污染敏感性实验必须分栏，不能将二者混称为同一个实验。

## 固定资源

| 文件 | 字节数 | MD5 | SHA256 | 状态 |
|---|---:|---|---|---|
| `clotho_audio_development.7z` | 4,541,582,263 | `c8b05bc7acdb13895bb3c6a29608667e` | `[MISSING]` | 等待远程完整校验 |
| `clotho_audio_validation.7z` | 1,260,701,425 | `7dba730be08bada48bd15dc4e668df59` | `[MISSING]` | 等待远程完整校验 |
| `clotho_captions_development.csv` | 1,336,762 | `d4090b39ce9f2491908eebf4d5b09bae` | `df2e5b92060b4bb23311f8b3a7f82241d900b9c4f62b0cc467ac2ce5e9c52886` | 本地验证完成 |
| `clotho_captions_validation.csv` | 367,649 | `5879e023032b22a2c930aaa0528bead4` | `fb0365506fe2dfcba9b7299daf7623a795abbd6ab9997a88ab0308e2fdfdbb88` | 本地验证完成 |
| `clotho_metadata_development.csv` | 830,797 | `170d20935ecfdf161ce1bb154118cda5` | `b054a8d9d0f88436e7cf6341c82a70e9e975c3b3ddd560d9b04f2cd5fdc75949` | 本地验证完成 |
| `clotho_metadata_validation.csv` | 224,803 | `2e010427c56b1ce6008b0f03f41048ce` | `066026ae1bc20277614ae9d4fffea085d959f0b5e40120751b8d3e717f5faa97` | 本地验证完成 |

字节数和 MD5 来自官方 Zenodo file metadata；四个小型 CSV 的 SHA256 是在本地真实下载并通过 MD5/字节数后计算的。两个大归档尚未在本地下载，所以不编造 SHA256。

## 元数据完整性

| 检查项 | development | validation |
|---|---:|---:|
| caption rows | 3,839 | 1,045 |
| metadata rows | 3,839 | 1,045 |
| caption/metadata 文件名集合 | 精确一致 | 精确一致 |
| captions/clip | 5 | 5 |
| 公共 loader 展开的音频文本对 | 19,195 | 5,225 |
| 空 caption | 0 | 0 |
| 单 clip 内重复 caption | 0 | 0 |
| 有效 FreeSound sound ID | 3,830 | 1,039 |

两个 metadata CSV 不是 UTF-8：development 和 validation 均包含原始 `0xC1` 字节，必须用严格 `ISO-8859-1` 解码。DATA-10 禁止 `errors="ignore"` 或 replacement 解码，以免静默改变来源字段。

官方文件名中有必须原样保留的前导空格：

- development：` Ambience Birds.wav`、` typical neighborhood in Porto.wav`
- validation：` e ieio ieai.wav`

任何 `.strip()`、自动重命名或扁平化都可能破坏 CSV—音频精确映射，因此只允许对 caption 文本去除外围空白，绝不修剪 `file_name`。

## 跨 split 重叠

- 精确文件名重叠 1 个：`City Ambience w_ Car Passing_1-2.wav`。
- 有效 FreeSound sound ID 重叠 2 个：`86163` 与 `130603`。
- `130603` 在两个 split 中使用不同的 HTML/转义文件名；`86163` 对应上述同名文件。
- 当前只证明 metadata 重叠；远程解压后 DATA-10 会计算重叠候选的逐文件 SHA256，判断是否为完全相同的音频字节。

这些记录来自官方 v2.1 release。校验器只报告，不删除、不改名，也不据此修改训练/验证候选集。

## DATA-10 可审计流程

入口：`bash scripts/run_data10_clotho_trainval_validation.sh`

1. 逐文件验证固定字节数和 MD5，并额外验证四个 CSV 的 SHA256。
2. 用 `7z l -slt` 在解压前审计全部成员；拒绝绝对路径、路径穿越、link、非 WAV 文件、重复路径和大小写冲突。
3. 分别解压到隔离的 staging 目录，成功且 WAV 数精确为 3,839/1,045 后才原子移动到 `extracted_trainval/{development,validation}`。
4. 已存在但没有 completion marker 的目录会被拒绝；脚本没有递归删除或覆盖路径。
5. 解码全部 4,884 个 WAV，核对 CSV/音频文件名集合，保留时长、采样率、声道、格式、subtype 与过滤状态。
6. canonical manifest 若内容相同则复用；若已有内容不同则拒绝覆盖。
7. 保存完整 stdout/stderr、退出码、Git commit/status、归档 listing/audit、来源 hash、统计 JSON 和 manifest hash。

预期输出：

- `logs/data10_clotho_trainval_validation_<timestamp>/`
- `${DATA_ROOT}/clotho_v2.1/manifests/clotho_development_manifest.jsonl`
- `${DATA_ROOT}/clotho_v2.1/manifests/clotho_validation_manifest.jsonl`

## 训练口径决策

在开始 `+Cl` 训练前保留两条互不覆盖的实验路线：

1. **公开代码路线**：development 训练、evaluation 早停；标记 `[CODE]`，并明确它存在 evaluation-set early-stopping contamination。
2. **无污染敏感性路线**：development 训练、official validation 早停；标记为替代/敏感性实验，不能写成论文 exact reproduction。

除非作者补充论文实际 split，否则不把任一路线标为 `[PAPER] exact`。
