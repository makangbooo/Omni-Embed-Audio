# Omni-Embed-Audio 复现资源需求

最后更新：2026-08-11

## 使用原则

只下载和保留 OEA 论文复现需要的模型、数据和运行工件。资源必须绑定 revision、
文件大小和可用的 SHA256/LFS 身份；模型或数据“目录存在”不等于资源完整。

## 软件环境

- Linux、Git、Bash、tmux 和 7-Zip。
- Conda 环境 `oea-repro`，依赖见 `environment.yml`、
  `requirements.txt` 和 `requirements-lock.txt`。
- Python/PyTorch/CUDA 组合必须先通过 `scripts/check_environment.py`、
  `scripts/validate_gpu_environment.sh` 和 BF16 smoke。
- 评测阶段需要 SoundFile/音频解码和 NumPy/JSON 工具。

## 官方 OEA 模型

需要三种 base backbone 及六个作者 checkpoint：

- Nemotron-3B base，OEA-Nemo3B-AC，OEA-Nemo3B-Cl；
- Qwen2.5-Omni-3B base，OEA-Qwen3B-AC，OEA-Qwen3B-Cl；
- Qwen2.5-Omni-7B base，OEA-Qwen7B-AC，OEA-Qwen7B-Cl。

固定 revision 和 checkpoint 身份见 `configs/checkpoints/` 与
`results/model_locks/`。远程 canonical 根目录为
`/home/jg525/models/oea`。不要删除 model lock 指向的 base model、tokenizer、
processor、LoRA 或双 projection-head 文件。

## 论文数据

| 数据 | 用途 | 必须保留的内容 |
|---|---|---|
| Clotho v2.1 | Table 2/3、UIQ、Negative UIQ | evaluation WAV、5-caption CSV、metadata、UIQ 对齐 manifest |
| AudioCaps v2 | Table 2/3、UIQ、Negative UIQ | test audio、captions、固定 metadata/manifests |
| MECAT `00A/test` | Table 2/3、UIQ、Negative UIQ | 848 public FLAC、六字段 metadata、public848 manifest |
| WavCaps | 训练数据审计和 leakage/provenance | 固定 metadata、过滤统计、blocklist/candidate artifacts |
| UIQ | Tables 4/12–15/17 | `data/UIQ/` 原始 JSONL 与 pairing audit |

远程 canonical 数据根目录为 `/home/jg525/datasets/oea`。具体准备步骤见
`docs/data_preparation.md` 和各数据审计文档。

## 基线模型

受控主矩阵需要：

- LAION-CLAP；
- M2D-CLAP；
- MGA-CLAP；
- Robust-CLAP；
- vanilla Nemotron-3B、Qwen2.5-Omni-3B、Qwen2.5-Omni-7B。

每个模型都必须保留 checkpoint、tokenizer/processor、源码 revision、
embedding config 和运行审计，不能只保留最终指标。

## GPU 与 CPU

### RTX 4090 24GB

适合大多数 3B 模型和 CLAP 基线的 smoke、分块 embedding 和受控效率实验。
Qwen7B 在较大 batch 下可能 OOM，必须使用小 batch 或 A100。单次全量生成的
时间取决于模型和数据集，正式脚本应在 smoke 后给出实测估计。

### A100 80GB

适合 Qwen7B、论文效率硬件边界和更大的 embedding batch。论文没有公开完整
的 timing/memory procedure，因此同型号 GPU 也不能自动解除 strict claim。

### CPU

下载、hash、解压、音频验证、NPZ 排名、表格生成和测试不需要 GPU。远程实例
曾只有两个可用 CPU 核，大文件 hash 和全量音频解码需要按小时估计。

## 存储

模型和原始数据需要百 GB 级共享空间；完整 embedding、排名和多次运行日志会
进一步增长。建议分别保留：

- `/home/jg525/models/oea`：官方模型与基线模型；
- `/home/jg525/datasets/oea`：论文数据；
- `/home/jg525/Omni-Embed-Audio/results/raw`：正式原始结果；
- `/home/jg525/Omni-Embed-Audio/logs`：正式运行日志；
- `/home/jg525/Omni-Embed-Audio/results/audits`：可提交的小型证据。

清理前应先列目录大小、确认 model lock/审计引用，再按精确目录名删除；不要对
`models/oea`、`datasets/oea`、`results/raw` 或 `logs` 做前缀不受控的递归清理。

## 不需要的资源

当前 OEA 论文复现不需要任何后续创新实验的专用语料、训练 checkpoint、
cache、reranker、adapter 或诊断日志。此类资源不应重新下载到上述 canonical
目录，也不应重新加入项目待办。
