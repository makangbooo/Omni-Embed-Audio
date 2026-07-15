# Omni-Embed-Audio 环境说明

最后更新：2026-07-15（Asia/Shanghai）

## 1. 已审计硬件

| 项目 | 结果 | 来源 |
|---|---|---|
| CPU 容器 | Ubuntu 22.04.5、glibc 2.35、GCC 11.4、约 503 GiB RAM；容器实际 `nproc=2` | 远程实测 CPU-01 |
| GPU 容器 | Ubuntu 22.04.5、glibc 2.35、GCC 11.4、约 1 TiB RAM；容器实际 `nproc=13` | 远程实测 GPU-01B |
| GPU | 1× NVIDIA A100-SXM4-80GB，MIG disabled，BF16-capable | 远程实测 GPU-01B |
| Driver | 580.95.05；`nvidia-smi` 报告最高 CUDA 13.0 | 远程实测 GPU-01B |
| CUDA Toolkit | `/usr/local/cuda-12.6`，NVCC 12.6.85，环境标记 12.6.3 | 远程实测 GPU-01B |
| NCCL | 系统 `libnccl.so.2`，环境标记 2.23.4-1 | 远程实测 GPU-01B |
| GPU fabric | A100 12 条 NVLink 均报告 25 GB/s；单卡实例无法验证 GPU–GPU 拓扑 | 远程实测 GPU-01B |
| 网络 | 9 个 `mlx5` 设备，200 Gb/s，Link layer Ethernet；多卡 NCCL/RoCE 尚未实测 | 远程实测 GPU-01B |
| limits | GPU 容器 memlock unlimited、stack 64 MiB、open files 1,024,000 | 远程实测 GPU-01B |
| 共享存储 | DPC 挂载，约 157 TB 可用；CPU/GPU 容器读取同一项目目录 | 远程实测 CPU-01/GPU-01B |

原始日志仅保留在远程 `logs/`，不进入普通 Git：

- `logs/system_info_cpu.txt`
- `logs/system_info_gpu_a100_20260715_151126.txt`

服务器名称和端口是临时的。实验身份以实验 ID、Git commit、时间、GPU UUID 和配置为准，不把 hostname 写进可复用配置。

## 2. 版本来源与决策

| 组件 | 候选版本 | 标记 | 依据 |
|---|---|---|---|
| Python | 3.11 | `[CODE]` | 官方 `requirements.txt` 注明来自 `.venv-oea311`；不使用 base 的 Python 3.13 |
| PyTorch/TorchVision/TorchAudio | 2.7.1/0.22.1/2.7.1 + cu126 | `[INFERRED]` | 官方仅要求 torch/torchaudio ≥2.5 并建议 cu124；服务器安装 CUDA 12.6，PyTorch 官方提供匹配的 cu126 wheel |
| Transformers | peeled commit `cb39f7d` | `[CODE]` | Qwen2.5-Omni 与 NVIDIA Omni-Embed 官方 model card 均指定 tag `v4.51.3-Qwen2.5-Omni-preview` |
| Tokenizers | 0.21.4 | `[INFERRED]` | 上述 Transformers commit 要求 `>=0.21,<0.22` |
| PEFT | 0.18.0 | `[CODE]` | 官方仓库要求 `peft>=0.18.0`；先固定最低公开版本 |
| Accelerate | 1.10.1 | `[INFERRED]` | 满足官方 `>=1.0.0` 和 PEFT/Transformers 要求，固定 2025 年稳定版本 |
| FlashAttention | GPU-02 不安装 | `[INFERRED]` | NVIDIA model card 推荐，但官方 OEA adapter 允许不指定；先验证 SDPA，避免把编译问题与核心环境混在一起 |

主要来源：

- 官方仓库 `requirements.txt`
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [Qwen2.5-Omni official repository](https://github.com/QwenLM/Qwen2.5-Omni)
- [Qwen2.5-Omni-3B model card](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- [NVIDIA Omni-Embed-Nemotron-3B model card](https://huggingface.co/nvidia/omni-embed-nemotron-3b)
- [Pinned Transformers commit](https://github.com/huggingface/transformers/tree/cb39f7dd5ba874ee1859b47283b08cd3a6ab5a0d)

## 3. 已发现的官方依赖冲突

1. `[CODE]` 根目录要求 `tokenizers>=0.22.0`，但官方 base-model card 指定的 `[CODE]` Transformers commit 明确要求 `tokenizers>=0.21,<0.22`。GPU-02 使用 0.21.4，不能直接执行未经修正的 `pip install -r requirements.txt`。
2. `[CODE]` 官方只给宽松下界，不能作为可审计 lock。`requirements-lock.txt` 当前只是直接依赖候选；GPU-02 成功后必须保存完整 `pip freeze --all` 和 conda export，再由本地审计后升级为最终 lock。
3. `[PAPER]` 使用 BF16/DDP；`[CODE]` trainer 尚未实现真实 DDP，且 autocast dtype 路径不明确。环境可用不代表训练实现已经符合论文。
4. `[MISSING]` 论文未给训练 GPU 数量。8×A100-80GB 是后续全量训练的资源候选，不是论文事实。

## 4. GPU-02 范围

GPU-02 只完成：

- 创建全新的 `oea-repro`；
- 安装 PyTorch cu126 与核心依赖；
- 检查 imports、CUDA、BF16、NCCL availability、音频解码/重采样；
- 保存完整解析后的依赖和日志。

GPU-02 不下载任何模型或数据，不安装 FlashAttention，不运行训练。若环境已存在，安装脚本会拒绝覆盖。
