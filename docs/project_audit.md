# Omni-Embed-Audio 第一阶段项目审计

## 1. 审计对象与版本

| 对象 | 固定版本 | 说明 |
|---|---|---|
| 论文 | ACL Anthology `2026.acl-long.1038` | 17 页；PDF SHA256 `E76BBD96...CFE38` |
| 官方代码 | `b261ad0743dbfac67cb6b20016fd951a366b3f35` | 首次公开版本；本地 tree SHA 与 GitHub tree SHA 均为 `39140092c86f37f4d52e321116b9971881c94dd4` |
| 本地工作分支 | `repro/oea-full` | 从上游 commit 精确重建；未修改 main |
| UIQ release | 13,053 rows | AudioCaps 4,530；Clotho 4,722；MECAT 3,801 |

## 2. 论文方法摘要

OEA 使用同一个具备原生音频理解能力的多模态 LLM 处理文本和音频。文本在 chat template 中使用 `query:` 前缀；论文称音频使用 `passage:` 前缀。两种模态都从最后一层 hidden states 做 attention-mask-aware mean pooling，再进入模态专用的无偏置线性投影、dropout、LayerNorm 和 L2 normalization，得到 512 维 embedding。backbone 冻结，只训练 LoRA 和两个投影头。

训练按 WavCaps 初始对齐、AudioCaps v2 caption retrieval、可选 Clotho v2 追加训练三个阶段进行。目标为温度 0.07 的 symmetric InfoNCE。论文报告三种 backbone、六个最终变体，并在 AudioCaps、Clotho、MECAT 上评估 caption T2A/T2T、四种正向 UIQ 和否定查询。

## 3. 参数溯源与冲突

| 项目 | 论文 | 官方代码 | 审计结论 |
|---|---|---|---|
| 文本前缀 | `[PAPER] query:` | `[CODE] query:` | 一致 |
| 音频前缀 | `[PAPER] passage:` | `[CODE]` 参数存在，但 `_build_audio_messages()` 明确不使用 | 冲突；必须做 checkpoint A/B smoke |
| 音频 | `[PAPER] 16 kHz mono` | `[CODE]` processor sampling rate，loader 转 mono/重采样 | 基本一致 |
| pooling | `[PAPER]` last hidden mean pooling | `[CODE]` attention mask mean pooling | 一致 |
| projection | `[PAPER]` Linear(no bias) → Dropout(0.1) → LayerNorm → L2，512d | `[CODE]` `ProjectionHead` 完全对应 | 一致 |
| LoRA | `[PAPER]` r=16, α=32, dropout=0.05；Q/K/V/O | `[CODE]` 同数值；模块名还含 `qkv,out_proj` | 数值一致；模块集合按架构兼容扩展 |
| backbone 冻结 | `[PAPER]` 是 | `[CODE]` adapter 先 `requires_grad_(False)`，PEFT 后仅 LoRA 可训练 | 一致 |
| loss | `[PAPER]` symmetric InfoNCE, τ=0.07 | `[CODE]` 默认一致 | 一致 |
| optimizer/LR | `[PAPER]` AdamW，3e-4 或 5e-4 | `[CODE]` trainer 默认 1e-4；WavCaps shell 为 5e-4 | stage 对应关系 `[MISSING]`，默认冲突 |
| precision | `[PAPER]` BF16 | `[CODE]` model 默认 BF16；autocast 未指定 dtype，GradScaler 路径更像 FP16 | 训练精度口径不确定 |
| DDP | `[PAPER]` PyTorch DDP | `[CODE]` 无 distributed 初始化、sampler 或 DDP wrapper | 缺失 |
| early stopping | `[PAPER]` validation R@10 | `[CODE]` 按 validation loss 保存/早停，只记录 R@10 | 冲突 |
| scheduler/warmup | `[MISSING]` | `[MISSING]` | 未公开/未实现 |
| gradient clipping | `[MISSING]` | `[MISSING]` | 未公开/未实现 |
| random seed | `[MISSING]` | `[CODE]` 训练未设置；WavCaps split seed=1337 | 训练不可重复 |
| epoch | `[MISSING]` | `[CODE]` WavCaps=15、AudioCaps=5、Clotho=10 | 只能视作代码默认 |
| WavCaps val | `[MISSING]` | `[CODE]` 每 source 按 MD5 划 1%，seed=1337 | 代码推定，非论文明确 |
| WavCaps ≤31s | `[PAPER]` 强制 | `[CODE]` manifest 支持参数但默认 `None` | 必须显式修正 |
| 泄漏 blocklist | `[PAPER]` 已应用 | `[CODE]` 无生成/训练过滤逻辑 | 缺失 |
| hard-negative Top-K | `[PAPER]` K=20 | `[CODE]` MGA pipeline=50；LAION 替代脚本=20 | 主路径冲突，替代模型也不同 |
| semantic model | `[PAPER]` BGE-large-en-v1.5 | `[CODE]` `BAAI/bge-large-en-v1.5`，阈值 0.7 | 模型一致；阈值仅代码给出 |

## 4. 论文表格与图

- 图 1 是表 2–4/17 的派生摘要；无绘图脚本。
- 图 2 可由 adapter、trainer 和 projection head 代码进行结构核验，但公开代码忽略音频 `passage:` 前缀。
- 图 3 定义了 HNSR@k 的关键条件；官方代码没有实现或单元测试。
- 主表 2、3 需要 13 个模型 × 3 个数据集；仓库只提供部分 adapter，统一入口不能跑完整矩阵。README 把 Hydra `key=value` 语法传给 argparse CLI，命令会直接报错；单独执行 `eval_hydra.py` 又不会加载 OEA LoRA/projection checkpoint。
- 表 4、12–15 所需四类正向 UIQ 已发布，但 loader schema 不匹配。
- 表 17 不能从当前 negative release 直接复现，因为 HN audio ID 没有发布。
- 表 5/16 只报告 A100-SXM4-80GB 结果，没有 warmup、batch、重复次数或 benchmark 脚本。

## 5. 发布数据一致性检查

| 数据 | rows | unique `audio_id` | 发现 |
|---|---:|---:|---|
| AudioCaps 每类正 UIQ | 975 | 975 | 与论文一致 |
| AudioCaps Negative | 630 | 255 | 同一 target 存在多条 query；无 HN ID |
| Clotho 每类正 UIQ | 1,045 | 1,045 | 与论文一致；正数据 ID 带扩展名，negative 多为 stem |
| Clotho Negative | 542 | 247 | 同上；无 HN ID |
| MECAT 每类正 UIQ | 848 | 848 | 与论文正文 847 不一致 |
| MECAT Negative | 409 | 176 | 无 HN ID |

全部 13,053 行均有非空 query，`source_model` 均为 `gpt-5.1`。三个 negative 文件的全部 1,581 行都只有 `negative_captions`，没有 `hard_negative_audio_id` 或等价字段。

## 6. 官方 checkpoint 元数据核验

| Repo | 固定 revision | 实际 checkpoint | 大小（decimal GB） |
|---|---|---|---:|
| `JudeJiwoo/OEA-Nemo3B-AC` | `8ed66aa77bc6f2001b807b5d2bd3e60503d89535` | `step_400_best.pt` | 9.47 |
| `JudeJiwoo/OEA-Nemo3B-Cl` | `9588912298afca0b11f5895b864ae28083f35022` | `step_450_best.pt` | 9.47 |
| `JudeJiwoo/OEA-Qwen3B-AC` | `f6c4b3b86385fd7ecbe3bacf45548a2259af8db4` | `step_350.pt` | 9.47 |
| `JudeJiwoo/OEA-Qwen3B-Cl` | `54ccd008d4a1340d2a1f8edcd5dd0e82c61367a4` | `step_40.pt` | 9.47 |
| `JudeJiwoo/OEA-Qwen7B-AC` | `f44f247020a7192affe6927db91d0778d33b9791` | `step_300.pt` | 17.94 |
| `JudeJiwoo/OEA-Qwen7B-Cl` | `30c6e97cfdf451b1948013d2839befe0c3022c46` | `step_330.pt` | 17.94 |

六个 checkpoint 合计 73.75 GB；三个 base-model 权重合计约 43.74 GB。模型卡统一写 `step_40.pt` 并称 base weights 不在 checkpoint 中，但实际 checkpoint 大小接近完整 backbone；下载与加载策略必须按真实文件处理。

## 7. 结论与复现边界

核心结论可以尝试复现，但公开发布不足以直接重跑论文全部表格。第一优先级不是训练，而是：固定环境；修复官方 checkpoint 加载/文件名；建立三个数据集的 canonical manifest；补 UIQ schema adapter 和 metric unit tests；在一个 3B checkpoint 上完成端到端官方权重评测。只有该闭环通过后，才进入单个 3B 训练链。

训练数字目前只能计划为“近似复现”，直到获得 AudioCaps v2 91,256-sample manifest、实际 DDP/global-batch 配置、stage-specific LR、精确 blocklists、HN pairing 和随机种子。
