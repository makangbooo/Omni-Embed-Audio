# MODEL-03/04 资源完整性审计

## 目的

MODEL-03 和 MODEL-04 的大文件下载曾多次因 CAS/Xet 连接中断而保留 partial。仅查看目录大小、README 或零字节 `.lock` 文件不能证明模型完整；本项目也不会为了重新开始下载而删除 partial。

入口：

```bash
bash scripts/run_model03_model04_audit.sh
```

该入口只读取远程 revision metadata 和本地文件，不调用 `snapshot_download`，不创建、修复、移动或删除模型文件。

## 固定审计范围

- `[CODE]` MODEL-03：Nemotron-3B base、OEA-Nemo3B-AC、OEA-Nemo3B-Cl 的三个不可变 revisions。
- `[CODE]` MODEL-04：Qwen2.5-Omni-7B base、OEA-Qwen7B-AC、OEA-Qwen7B-Cl 的三个不可变 revisions。
- 每个 asset 的目标目录和 revision marker 必须与资源 manifest 完全一致。
- 本地选定文件集合必须与固定 revision 的远程选定文件集合完全一致；缺失和额外文件均单独报告。
- 每个文件验证字节数；LFS 文件验证远程 LFS SHA256，非 LFS 文件验证 Git blob SHA-1，并额外记录本地 SHA256。
- `.cache/` 不作为模型快照文件；其中持久存在的 `.lock` 不判失败，真实 `*.incomplete` 会被列出并使 asset 状态为 `incomplete`。
- symlink、revision marker 缺失/不一致、额外文件、尺寸错误或内容 hash 错误判为 `failed`，不会自动修复。

## 状态与退出码

| 报告状态 | 退出码 | 含义 | 后续动作 |
|---|---:|---|---|
| `complete` | 0 | 六个 asset 的完整选定快照均与固定 revision 一致 | 可进入 checkpoint 结构审计/评测 |
| `incomplete` | 2 | 至少有缺失文件或 `.incomplete`，且没有已证明的内容/来源错误 | 只对相应 asset 断点续传 |
| `failed` | 1 | 网络 metadata 审计失败，或 marker/文件集合/内容不一致 | 保留报告，先分析根因；不得直接删除或重下 |

单个 asset 同样使用 `complete`、`incomplete`、`failed`。整体状态不能用目录总大小或某一个 checkpoint 的完整性替代。

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

## 运行资源

- 服务器：CPU 即可；GPU 被显式禁用。
- 环境：`oea-repro`。
- 网络：仅访问 Hugging Face revision/file metadata，不下载模型内容。
- 本地读取：约 86.7 GB `[INFERRED]` 的 MODEL-03/04 已存在文件；具体读取量取决于 partial 状态。
- 预计时间：完整时约 10–45 分钟 `[INFERRED]`，主要取决于共享存储顺序读速度和 SHA256 计算。
- 输出目录：仓库 `logs/`，已由 `.gitignore` 排除。
- 覆盖/删除风险：无；报告使用新的时间戳目录。
