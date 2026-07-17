# 官方模型资源完整性审计

## 目的

MODEL-03 和 MODEL-04 的大文件下载曾多次因 CAS/Xet 连接中断而保留 partial。仅查看目录大小、README 或零字节 `.lock` 文件不能证明模型完整；本项目也不会为了重新开始下载而删除 partial。

入口：

```bash
bash scripts/run_model03_model04_audit.sh
```

该入口只读取远程 revision metadata 和本地文件，不调用 `snapshot_download`，不创建、修复、移动或删除模型文件。

六个正式 OEA 变体优先使用按变体入口：

```bash
bash scripts/run_official_oea_model_resource_audit.sh oea_qwen3b_cl
```

该入口只审计所选变体的一个基础模型和一个官方 checkpoint。两者完全由
`configs/checkpoints/official_oea_checkpoints.json` 解析，不维护第二份 case
映射。特别地，`oea_qwen3b` 会从 MODEL-01 manifest 选择 Qwen3B base，
同时从 MODEL-02 manifest 选择 AC checkpoint；同一 manifest 内的无关
checkpoint 不会被重复扫描。可用 ID 为：

```text
oea_nemo3b
oea_nemo3b_cl
oea_qwen3b
oea_qwen3b_cl
oea_qwen7b
oea_qwen7b_cl
```

按变体入口要求干净 Git worktree，在报告 `scope` 中固定 registry 本身的
SHA256、完整解析结果、两个 manifest 和两个 asset 名称。这样生成的报告
可直接作为可移植模型锁的输入。原 `run_model03_model04_audit.sh` 仍保留，
用于一次性判断 MODEL-03/04 六个下载任务分别需要续传还是已经完成。

## 固定审计范围

- `[CODE]` MODEL-03：Nemotron-3B base、OEA-Nemo3B-AC、OEA-Nemo3B-Cl 的三个不可变 revisions；两个 checkpoint 的实际文件分别为 `step_400_best.pt` 与 `step_450_best.pt`。
- `[CODE]` MODEL-04：Qwen2.5-Omni-7B base、OEA-Qwen7B-AC、OEA-Qwen7B-Cl 的三个不可变 revisions；两个 checkpoint 的实际文件分别为 `step_300.pt` 与 `step_330.pt`。
- 每个 asset 的目标目录和 revision marker 必须与资源 manifest 完全一致。
- 本地选定文件集合必须与固定 revision 的远程选定文件集合完全一致；缺失和额外文件均单独报告。
- 每个文件验证字节数；LFS 文件验证远程 LFS SHA256，非 LFS 文件验证 Git blob SHA-1，并额外记录本地 SHA256。
- `.cache/` 不作为模型快照文件；其中持久存在的 `.lock` 不判失败，真实 `*.incomplete` 会被列出并使 asset 状态为 `incomplete`。
- symlink、revision marker 缺失/不一致、额外文件、尺寸错误或内容 hash 错误判为 `failed`，不会自动修复。
- checkpoint 的精确文件名、字节数与 LFS SHA256 已写入资源 manifest；README 中“一律 `step_40.pt`”的说法不作为文件选择依据。

## 状态与退出码

| 报告状态 | 退出码 | 含义 | 后续动作 |
|---|---:|---|---|
| `complete` | 0 | 六个 asset 的完整选定快照均与固定 revision 一致 | 可进入 checkpoint 结构审计/评测 |
| `incomplete` | 2 | 至少有缺失文件或 `.incomplete`，且没有已证明的内容/来源错误 | 只对相应 asset 断点续传 |
| `failed` | 1 | 网络 metadata 审计失败，或 marker/文件集合/内容不一致 | 保留报告，先分析根因；不得直接删除或重下 |

单个 asset 同样使用 `complete`、`incomplete`、`failed`。整体状态不能用目录总大小或某一个 checkpoint 的完整性替代。

完整性审计通过后，使用 `scripts/run_official_checkpoint_preparation.sh`
执行逐 checkpoint 的 weights-only 结构检查与非覆盖推理权重提取；详见
`docs/official_checkpoint_preparation.md`。完整快照通过不自动证明内部
LoRA/projection 结构正确，两层审计证据必须分别保留。

## 审计工件

每次运行写入：

```text
logs/model03_model04_audit_<timestamp>/
├── command.sh
├── git_commit.txt
├── git_status.txt
├── hostname.txt
├── audit_exit_code.txt
├── wrapper_exit_code.txt
├── stdout.log
├── stderr.log
└── model_resource_audit.json
```

`model_resource_audit.json` 保存远程文件清单、逐文件本地 hash、远程 LFS/Git blob 匹配结果、missing/extra/incomplete 文件和完整错误。大型模型文件不进入 Git；后续只提交小型摘要与报告 hash。

按变体入口使用目录：

```text
logs/official_model_resource_audit_<variant>_<timestamp>/
```

并保存同类工件、审计前后内存/磁盘信息，以及精确的 variant scope。

## 运行资源

- 服务器：CPU 即可；GPU 被显式禁用。
- 环境：`oea-repro`。
- 网络：仅访问 Hugging Face revision/file metadata，不下载模型内容。
- 本地读取：约 86.7 GB `[INFERRED]` 的 MODEL-03/04 已存在文件；具体读取量取决于 partial 状态。
- 预计时间：完整时约 10–45 分钟 `[INFERRED]`，主要取决于共享存储顺序读速度和 SHA256 计算。
- 按变体读取量：Nemo3B 约 18.9 GB、Qwen3B 约 21.4 GB、Qwen7B 约 40.3 GB `[CODE][INFERRED]`；实际值由报告逐文件累计，预计 10–60 分钟。
- 输出目录：仓库 `logs/`，已由 `.gitignore` 排除。
- 覆盖/删除风险：无；报告使用新的时间戳目录。
