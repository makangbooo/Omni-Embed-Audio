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
| `OEA-Nemo3B-AC` | `8ed66aa77bc6f2001b807b5d2bd3e60503d89535` | `[CODE] step_400_best.pt`；9,466,826,153 bytes；LFS SHA256 已固定 | 9.47 GB |
| `OEA-Nemo3B-Cl` | `9588912298afca0b11f5895b864ae28083f35022` | `[CODE] step_450_best.pt`；9,466,826,217 bytes；LFS SHA256 已固定 | 9.47 GB |
| `OEA-Qwen7B-AC` | `f44f247020a7192affe6927db91d0778d33b9791` | `[CODE] step_300.pt`；17,940,602,533 bytes；LFS SHA256 已固定 | 17.94 GB |
| `OEA-Qwen7B-Cl` | `30c6e97cfdf451b1948013d2839befe0c3022c46` | `[CODE] step_330.pt`；17,940,602,661 bytes；LFS SHA256 已固定 | 17.94 GB |

六个 OEA checkpoint 共 73.75 GB，三个 base 权重共约 43.74 GB，合计 117.49 GB（109.42 GiB），未含缓存临时文件。建议预留 160 GB。

当前并行批次采用完整不可变快照，不依据 README 中有冲突的 `step_40.pt` 说法猜测：MODEL-03 约 28.4 GB、MODEL-04 约 58.3 GB。六个 OEA 原始 checkpoint 的实际远端文件名、字节数与 LFS SHA256 已固定在资源 manifest 和 `configs/checkpoints/official_oea_checkpoints.json`。

`[CODE]` 已提供 `scripts/run_model03_model04_audit.sh` 做只读完成性判定：完整比对固定 revision 文件集合、字节数、LFS SHA256、非 LFS Git blob ID、revision marker 与残留 `.incomplete`。它不会下载或修改模型；详见 `docs/model_resource_audit.md`。

完整性审计通过后，`scripts/run_official_checkpoint_preparation.sh` 对任一变体执行 weights-only/FakeTensorMode 结构审计，并以实测 LoRA 结构生成非覆盖的 inference-only 权重。其余五个评测配置必须等待各自派生权重 SHA256，不提前猜测；详见 `docs/official_checkpoint_preparation.md`。

## P3：数据

### WavCaps

- `[CODE]` 已固定公共 `cvssp/WavCaps` revision `0930ec11ded28fa0eaa910fde2f6fc3538acbeac`。DATA-08 只下载 8 个 metadata/官方 blacklist 文件，共 176,863,095 bytes；完整仓库（含音频）约 819.51 GB。
- `[CODE]` 固定元数据共有 403,050 条：AudioSet_SL 108,317、BBC 31,201、FreeSound 262,300、SoundBible 1,232。
- `[PAPER]` 报告长度 `<=31` 秒后为 275,618；公共元数据按该条件为 275,691，只有 `[INFERRED]` `0 < duration < 31` 才精确得到 275,618。论文精确 filtered manifest 仍为 `[MISSING]`。
- DATA-09 已精确复现 173 个 AudioCaps test 重叠和 638 个 Clotho evaluation 文件名重叠。638 个文件名对应 1,017 个 WavCaps 候选（64 个文件名存在歧义），因此 Clotho 保守 blocklist 是 `[INFERRED]`，不是论文未公开的原始 blocklist。
- `[MISSING]` 论文 validation split、精确去泄漏 blocklist、重复文件名消歧规则和去泄漏后的最终训练数量。metadata-only 保守重建为 275,062 条，不能标为 exact。
- 当前先在远程 CPU 侧复算 176.86 MB metadata；训练前才需要下载/准备全量 WavCaps 音频。
- 目录要求：`datasets/wavcaps/{metadata,audio}`；manifest 写入 `data/manifests/wavcaps/`，blocklist 写入 `data/blocklists/`。
- 详见 `docs/wavcaps_data_audit.md`。

### AudioCaps v2

- `[CODE]` 已固定官方 AudioCaps 2.0 commit `d004db3ea1b01cf4fd0347dd8d27db90cadc8809` 的 `dataset2.0/{train,val,test}.csv`，DATA-06/07 校验 MD5、SHA256、bytes、schema、计数及 UIQ 对齐。
- `[PAPER][CODE]` 论文与官方 README 均报告 91,256 train；`[CODE]` 公共 OEA loader 实际只能得到 91,254 个有效记录，原因是 3 个 bare-CR 孤立 caption 片段和 2 个 quoted 多行 caption。
- `[INFERRED]` 仅修复 3 个 caption 尾部仍为 91,254 条；`[MISSING]` 论文的有效 91,256-row manifest 或另外 2 条记录。
- test 已固定为 975 clips × 5 captions；四类正 UIQ 的 ID 和去重 captions、630 条 negative 的原 captions 均与官方 CSV 精确对齐。
- 当前真正需要用户提供的是官方音频下载权限/文件。官方 README 指向 `https://forms.gle/2wF54Y1Ft2LtPdhW8`；若只能从 YouTube 重建，必须逐条报告下架缺失，不能缩小候选集冒充论文口径。
- 详见 `docs/audiocaps_v2_data_audit.md`。

### Clotho v2.1

- 官方 Zenodo record `4783391`：development archive 4,541,582,263 bytes、validation archive 1,260,701,425 bytes、evaluation archive 1.2 GB，metadata/captions 约 3.3 MB，总下载约 7.1 GB。
- `[PAPER]` 追加训练为 3,839 clips；evaluation 为 1,045 clips。
- DATA-02 已在远程完整验证 v2.1 evaluation：1,045/1,045 音频成功解码、每条 5 captions、四类正 UIQ ID 全部精确对齐；manifest MD5 `253c1b275e3618fa94750150d7962da5`。
- DATA-10 已在本地完整审计 development/validation 的四个 CSV：3,839/1,045 条、各 5 captions，caption/metadata 文件名集合精确一致；严格 metadata 编码为 ISO-8859-1。两个音频归档及 4,884 WAV 的远程校验待 DATA-03 下载完成。
- `[CODE]` 公开 `+Cl` launcher 用 development 训练、evaluation 早停；`[PAPER]` 只写 validation R@10，未命名 split。论文真实早停 split 为 `[MISSING]`，训练时必须把公开代码路线与 official-validation 敏感性路线分开报告。
- `[INFERRED]` 论文只写 v2；当前使用修复后的 v2.1 并在报告标记版本差异。详见 `docs/clotho_trainval_data_audit.md`。

### MECAT

- 官方数据仓库：`mispeech/MECAT-Caption` revision `be4a24c3f7309d74208e08a7cce49e72cb7a5834`；官方实现仓库：`xiaomi-research/mecat` commit `a004949d58e86e2ee56baa879607ec2109cfcc46`。
- 当前只需下载 `00A/test_0000-0000000.tar.gz`：173,168,424 bytes，LFS SHA256 `644cf75e2509c633452a18e36c41b285a317c6cbc06198d7dfe406c5aa5122c4`；无需下载约 16 GB 的其他域。
- `[CODE]` 官方 `00A/test` 和四个 OEA 正向 UIQ 文件均为 848 IDs；`[PAPER]` 评测写 847 auto-captioned pairs。
- DATA-11 在固定 WavCaps AudioSet_SL metadata 中找到 4 个 MECAT 同源 YouTube video candidates；该结果仅为 `[CODE][INFERRED]` provenance，不证明时间片/音频重复。远程 canonical manifest 复算等待 DATA-05/08。
- `[MISSING]` 论文使用的精确 847-row manifest、排除的 1 条样本、T2A/T2T caption 字段/组合。
- DATA-04/05 可以完成公开 848 条数据的下载、解压、解码、六字段保存和 UIQ ID 精确对齐；严格 847 条主表在作者提供缺失信息前保持 BLOCKED。详见 `docs/mecat_data_audit.md`。

### SQuTR（SpeechXBT/OEA-5 目标语料）

- `[CODE]` 官方数据仓库为 `SLLMCommunity/SQuTR`，DATA-12 固定 revision `2f1b041e2e98e0d28ed68fbcf22126ef247eb719`。
- `[CODE]` 官方仓库只发布一个 `source_data.zip`；精确大小为 21,069,841,248 bytes，LFS SHA256 为 `8956bf938de3f9ce168a1e7daf2ff61b0b7fe603fa5c3d7dc6a4314617c6997c`。
- `[PAPER][CODE]` 数据包含 37,317 个唯一查询、四种声学条件下共 149,268 个实例和约 190.4 小时音频；六个子集为 FiQA、NQ、HotpotQA、MedicalRetrieval、DuRetrieval、T2Retrieval。
- DATA-12 已在 CPU 服务器完成：运行 `data12_squtr_download_20260726_130854` 从 16,524,500,992 bytes 断点续传，最终大小和 SHA256 均精确匹配，退出码为 0；本步未解压。
- 默认目录为 `/home/jg525/datasets/oea/squtr/source/source_data.zip`；DATA-13A 实测解压后成员总大小为 28,422,366,590 bytes。DATA-13B 启动门禁要求至少 40,000,000,000 bytes 空闲，后续 manifest、冻结文本 embedding 和索引阶段建议总计预留 100 GB。
- `[CODE]` 官方代码仓库固定 commit `cc3fb31fc0dc44fef3a44c569b344516bbaee79c`；README 将 `queries_with_audio_*.jsonl` 放在子集根目录，而官方 `omni_emb.py` 将 `query_file` 拼接到声学条件目录。
- DATA-13A 已在 CPU 服务器完成：真实归档确认所有 42 个 JSONL 均存在、query metadata 位于子集根目录、qrels 位于 `qrels/test.jsonl`；149,268 个 WAV 的六子集四条件计数全部匹配，未解压且未做 CRC。
- DATA-13B 使用 CPU 逐成员 CRC 解压并验证实际 JSONL schema、corpus/query/qrels ID 闭包、文档单位、查询集合、SNR/noise 元数据和 149,268 个 WAV 首尾帧。预计 2–8 小时、0 GPU、额外约 28.42 GB 解压空间和少量 manifest/log 空间；可从工具自有 `.data13b.part` 恢复，但任何不匹配的最终文件都会拒绝覆盖。

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
| Vanilla backbones | 三个 base revision、base-only 锁流水线和 Clotho 正式生成器已固定 | 真实 base 锁尚未远程生成/提交；GPU fixture 尚未运行；AudioCaps/MECAT 候选集仍缺 |

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
| 官方权重评测 | Qwen3B 已实测可用 1×RTX 4090 24GB；A100-80GB 用于论文效率硬件对齐 | Qwen3B-Cl 5/25 smoke 峰值 allocated 9.14 GiB；7B 尚无本项目真实 smoke，不能据论文 18.3 GB 直接承诺 4090 可运行 |
| 第一个 3B 训练闭环 | 4–8×A100，80GB 优先 | `[INFERRED]` 公开 trainer 无 DDP，需先补齐；论文训练 GPU 数 `[MISSING]` |
| 7B 全量训练 | 8×A100-80GB 优先 | `[INFERRED]` 避免小 micro-batch 和 checkpoint 峰值 |
| smoke/单元测试 | 1×4090 或 1×A100 | 64–256 样本、单卡 |
| 官方权重与基线评测 | 约 10–30 A100 GPU-hours | `[INFERRED]` 可复用 audio/text embeddings；不含调试 |
| 一个 3B 三阶段链 | 约 200–600 A100 GPU-hours | `[INFERRED]` global batch/步骤和 DDP 配置未公开，范围较宽 |
| 三 backbone 全训练 | 约 1,500–4,000 A100 GPU-hours | `[INFERRED]` 9 个 stage run、早停与并行方式未知 |
| 项目墙钟时间 | 4–8 周 | 数据准备/审计 1–2 周，评测 1 周，训练与故障复盘 2–5 周；多服务器并行可缩短 |

任何超过 30 分钟或大量占盘的具体命令，在执行前重新给出 GPU 数、显存、时间、磁盘、输出目录与覆盖风险；本表不是执行授权。
