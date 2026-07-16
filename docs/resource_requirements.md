# Omni-Embed-Audio 资源需求清单

## 使用原则

- 本文件列出全项目资源，但“当前需用户提供”只包含 P0。数据、模型与 API 不会在项目审计阶段下载。
- 模型和数据一律固定 revision/checksum；无法固定的资源标为 `[MISSING]`，不先下载替代品冒充原资源。
- 普通 Git 只同步代码、配置、小型 metadata/metrics；音频、权重、checkpoint、embedding 和密钥只存远程数据盘/模型缓存。

## P0：现在需要（项目审计收尾与下一阶段入口）

### 1. GitHub 写入目标

- 用户账号下的官方仓库 fork URL，作为本地 `origin`。
- 本地 GitHub CLI `gh` 安装并登录；当前机器没有 `gh`。
- 当前终端到 `github.com:443` 的 `git clone` 两次超时，但 `api.github.com` 与 `codeload.github.com` 可访问。若 HTTPS Git 仍失败，下一步改用 `ssh.github.com:443`，不使用不明镜像。
- 当前本地代码已精确固定上游 `b261ad0743...` 并创建 `repro/oea-full`；尚未 commit/push 审计修改。

### 2. 一台候选 A100 服务器的只读系统信息

- 优先 A100-SXM4-80GB，因为论文效率表使用该硬件，且官方 7B checkpoint 加载过程可能出现额外显存峰值。
- 需要 GPU 型号/数量/显存、Driver、CUDA、GCC、conda、Python、CPU/RAM、磁盘、网络连通性。
- 若 A100 只有 40GB，也请如实返回；环境和 batch size 会据此调整。

## P1：环境通过后，首个 checkpoint smoke/eval

推荐先跑 `OEA-Qwen3B-Cl` 的 5 个随仓库样例，再跑 `OEA-Qwen3B-AC` 的 AudioCaps smoke。当前不下载，待环境检查通过后再给出带 revision 的命令。

| 资源 | 固定 revision | 文件/体量 | 访问 |
|---|---|---|---|
| `Qwen/Qwen2.5-Omni-3B` | `f75b40e3da2003cdd6e1829b1f420ca70797c34e` | repo 11.99 GB；weights 11.97 GB | 公共 HF |
| `JudeJiwoo/OEA-Qwen3B-Cl` | `54ccd008d4a1340d2a1f8edcd5dd0e82c61367a4` | `step_40.pt`, 9.47 GB | 公共 HF |
| `JudeJiwoo/OEA-Qwen3B-AC` | `f6c4b3b86385fd7ecbe3bacf45548a2259af8db4` | `step_350.pt`, 9.47 GB | 公共 HF |
| 示例音频 | 随代码 `b261ad0` | 5 个 Clotho wav；约 12.7 MB | 已在 repo |

预计首轮模型缓存 32–40 GB（含 HF 临时文件与版本缓存）；1×A100/4090；smoke 10–30 分钟，完整单模型三数据集评测约 1–3 GPU 小时 `[INFERRED]`。

## P2：六个官方 OEA checkpoint 与 base models

| 资源 | revision | 实际权重文件 | 大小 |
|---|---|---|---:|
| `nvidia/omni-embed-nemotron-3b` | `865db1bb57e369a85357cf114cbd6b3c5322d19d` | 多个 safetensors | repo 9.42 GB |
| `Qwen/Qwen2.5-Omni-7B` | `ae9e1690543ffd5c0221dc27f79834d0294cba00` | 多个 safetensors | repo 22.38 GB |
| `OEA-Nemo3B-AC` | `8ed66aa77bc6f2001b807b5d2bd3e60503d89535` | `[MISSING]` 下载固定 revision 完整快照后审计 | 约 9.47 GB `[INFERRED]` |
| `OEA-Nemo3B-Cl` | `9588912298afca0b11f5895b864ae28083f35022` | `[MISSING]` 下载固定 revision 完整快照后审计 | 约 9.47 GB `[INFERRED]` |
| `OEA-Qwen7B-AC` | `f44f247020a7192affe6927db91d0778d33b9791` | `[MISSING]` 固定 model card 与旧资源记录的 step 名称冲突 | 约 17.94 GB `[INFERRED]` |
| `OEA-Qwen7B-Cl` | `30c6e97cfdf451b1948013d2839befe0c3022c46` | `[MISSING]` 下载固定 revision 完整快照后审计 | 约 17.94 GB `[INFERRED]` |

六个 OEA checkpoint 共 73.75 GB，三个 base 权重共约 43.74 GB，合计 117.49 GB（109.42 GiB），未含缓存临时文件。建议预留 160 GB。

当前并行批次采用完整不可变快照，不依据有冲突的文件名猜测：MODEL-03 约 28.4 GB、MODEL-04 约 58.3 GB，实际远端文件名、字节数与 LFS SHA256 将由下载时的 `download_manifest.json` 固定。

## P3：数据

### WavCaps

- `[PAPER]` 训练实际使用 275,618 samples，长度 ≤31 秒。
- `[CODE]` 公共 `cvssp/WavCaps` revision `0930ec11ded28fa0eaa910fde2f6fc3538acbeac` 的所有仓库文件合计 819.51 GB；原始统计为 403,050 clips。
- `[MISSING]` 论文使用的 275,618-row 精确 manifest、validation split、去泄漏后的最终数量、两个 blocklist 文件。
- 优先请求作者/用户已有的精确 filtered manifest；若没有，再从公共 WavCaps 全量构建。
- 目录要求：`datasets/wavcaps/{metadata,audio}`；manifest 写入 `data/manifests/wavcaps/`，blocklist 写入 `data/blocklists/`。

### AudioCaps v2

- `[PAPER]` 91,256 training samples；test 为 975 clips × 5 captions。
- `[CODE]` 路径假设 `AudioCaps/v2_meta_data/{train,val,test}.csv` 与 `AudioCaps/audiocaps_raw_audio/`。
- `[MISSING]` “v2”来源、revision、91,256 样本究竟是 clip 还是 caption rows、公开下载说明与 checksum。
- 在获得论文同版 metadata 前，不用普通 AudioCaps 替代并声称严格复现。

### Clotho v2.1

- 官方 Zenodo record `4783391`：development 4.5 GB、validation 1.3 GB、evaluation 1.2 GB，metadata/captions 约 3.3 MB，总下载 7.1 GB。
- `[PAPER]` 追加训练为 3,839 clips；evaluation 为 1,045 clips。
- 需要确认论文是 v2.0 还是已修复的 v2.1 音频文件；推荐使用 v2.1 并在报告标记版本差异。

### MECAT

- 官方仓库：`xiaomi-research/mecat`。
- `[PAPER]` 评测 847 auto-captioned pairs；`[CODE]` UIQ 正向文件有 848 IDs。
- `[MISSING]` 论文使用的精确 847-row manifest、音频映射、caption 字段与排除的 1 条样本。
- 在得到 manifest 前，MECAT 相关主表保持 BLOCKED。

### UIQ 与 hard negatives

- UIQ 文本 13,053 行已随 repo 提供，无需另行下载。
- `[MISSING]` 1,581 条 negative queries 的 hard-negative audio ID；当前只有 negative captions。
- `[MISSING]` 人工复核后的 target/HN pairing 文件、移除记录与审计人员结论。
- 若作者不能提供，将用 captions 精确匹配/音频 ID 重建，结果必须标记 `[INFERRED]`，不能称 exact reproduction。

## P4：基线

| 基线 | 代码现状 | 仍需确认 |
|---|---|---|
| LAION-CLAP | adapter + Hydra config | `[MISSING]` 论文 checkpoint 的精确文件/revision；config 的 `ckpt_path=null` |
| Robust-CLAP | adapter 文件存在 | 无 config/统一 CLI；外部 repo、旧 torchlibrosa、checkpoint revision `[MISSING]` |
| MGA-CLAP | adapter +占位路径 | `mga-clap.pt` 的官方下载/revision/checksum `[MISSING]` |
| M2D-CLAP | portable adapter 存在 | 预期 `checkpoint-30.pth`，但无 config/统一 CLI/下载说明 |
| Vanilla backbones | base adapter 存在 | 必须固定与 OEA 相同 pooling/normalize；MECAT loader 缺失 |

在基线阶段开始前，先向作者仓库/模型卡核对 checkpoint，不能仅凭本地占位文件名下载近似模型。

## P5：闭源 API 与人工资源（最后阶段）

- GPT-5.1 API：用于五类 UIQ 再生成；论文参数 temperature=0.35、top-p=0.9、max tokens=256。
- Claude Opus 4.5 API：用于同一 75 样例的自动有效性评分；max tokens=256、0.5 秒限流。
- 9 名人工标注者、75 个固定样例 ID、音频和网页界面；论文原始 ratings/抽样 seed 未发布。
- 若模型版本已下线或 API 不可复现，替代模型实验必须与 paper-reported 分栏。

## 推荐环境（暂定，等待服务器 Driver 核验）

- conda 环境名：`oea-repro`。
- `[CODE]` Python 3.11（requirements 注释来自 `.venv-oea311`）。
- `[CODE]` CUDA 12.4 wheel 路线；torch/torchaudio 要求 ≥2.5.0，但未锁版本。
- `[CODE/BASE MODEL CARD]` Qwen/NVIDIA base model 要求 Hugging Face `v4.51.3-Qwen2.5-Omni-preview`；官方 repo 的 `transformers>=4.47.0` 过宽。
- `[INFERRED]` 初始兼容方案：Python 3.11 + CUDA 12.4 runtime + 与服务器 Driver 兼容的 PyTorch 2.5/2.6 +上述 transformers tag + PEFT 0.18；精确 lock 在最小导入/前向测试后生成。
- FlashAttention 2 先作为可选优化，不在第一个 CPU/import smoke 前强装；A100/4090 通过基础 SDPA 后再锁兼容版本。

## 总体资源预算

| 项目 | 建议 | 依据/置信度 |
|---|---|---|
| 磁盘（最低分阶段） | 约 1.5–2.0 TB | `[INFERRED]` 不长期保留全部压缩包/中间 checkpoint |
| 磁盘（完整可审计） | 2.5 TB；建议 3 TB | WavCaps 819.51 GB 压缩源 + 解压/筛选 + 117.49 GB OEA/base + 数据/基线/checkpoints |
| 官方权重评测 | 1×A100-80GB 优先；可评估 1×4090 | 论文 7B inference peak 18.3 GB，但当前 checkpoint 加载会产生额外峰值 |
| 第一个 3B 训练闭环 | 4–8×A100，80GB 优先 | `[INFERRED]` 公开 trainer 无 DDP，需先补齐；论文训练 GPU 数 `[MISSING]` |
| 7B 全量训练 | 8×A100-80GB 优先 | `[INFERRED]` 避免小 micro-batch 和 checkpoint 峰值 |
| smoke/单元测试 | 1×4090 或 1×A100 | 64–256 样本、单卡 |
| 官方权重与基线评测 | 约 10–30 A100 GPU-hours | `[INFERRED]` 可复用 audio/text embeddings；不含调试 |
| 一个 3B 三阶段链 | 约 200–600 A100 GPU-hours | `[INFERRED]` global batch/步骤和 DDP 配置未公开，范围较宽 |
| 三 backbone 全训练 | 约 1,500–4,000 A100 GPU-hours | `[INFERRED]` 9 个 stage run、早停与并行方式未知 |
| 项目墙钟时间 | 4–8 周 | 数据准备/审计 1–2 周，评测 1 周，训练与故障复盘 2–5 周；多服务器并行可缩短 |

任何超过 30 分钟或大量占盘的具体命令，在执行前重新给出 GPU 数、显存、时间、磁盘、输出目录与覆盖风险；本表不是执行授权。
