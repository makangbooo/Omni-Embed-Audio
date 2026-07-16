# Omni-Embed-Audio 环境说明

最后更新：2026-07-16（Asia/Shanghai）

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
| Transformers | 4.52.4 | `[INFERRED]` | model card 的 preview commit 无法导入 `[CODE]` 要求的 PEFT 0.18.0；PEFT 0.18.0 官方示例固定 4.52.4，且该版本包含 Qwen2.5-Omni 与 `modeling_layers` |
| Tokenizers | 0.21.4 | `[INFERRED]` | Transformers 4.52.4 要求 `>=0.21,<0.22` |
| PEFT | 0.18.0 | `[CODE]` | 官方仓库要求 `peft>=0.18.0`；先固定最低公开版本 |
| Accelerate | 1.10.1 | `[INFERRED]` | 满足官方 `>=1.0.0` 和 PEFT/Transformers 要求，固定 2025 年稳定版本 |
| FlashAttention | GPU-02 不安装 | `[INFERRED]` | NVIDIA model card 推荐，但官方 OEA adapter 允许不指定；先验证 SDPA，避免把编译问题与核心环境混在一起 |
| 7-Zip | 26.02 | `[INFERRED]` | 官方 Clotho evaluation 音频以 `.7z` 发布；CPU 预检确认服务器无系统 extractor，因此固定 conda-forge `7zip` 并通过 `7zz` 解压 |

主要来源：

- 官方仓库 `requirements.txt`
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [Qwen2.5-Omni official repository](https://github.com/QwenLM/Qwen2.5-Omni)
- [Qwen2.5-Omni-3B model card](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- [NVIDIA Omni-Embed-Nemotron-3B model card](https://huggingface.co/nvidia/omni-embed-nemotron-3b)
- [Pinned Transformers commit](https://github.com/huggingface/transformers/tree/cb39f7dd5ba874ee1859b47283b08cd3a6ab5a0d)
- [Transformers 4.52.4](https://github.com/huggingface/transformers/tree/v4.52.4)
- [PEFT 0.18.0 official example requirements](https://github.com/huggingface/peft/blob/v0.18.0/examples/int8_training/requirements.txt)

## 3. 已发现的官方依赖冲突

1. `[CODE]` 根目录要求 `tokenizers>=0.22.0`，但官方 base-model card 指定的 `[CODE]` Transformers commit 明确要求 `tokenizers>=0.21,<0.22`。GPU-02 使用 0.21.4，不能直接执行未经修正的 `pip install -r requirements.txt`。
2. `[CODE]` 官方只给宽松下界，不能作为可审计 lock。`requirements-lock.txt` 当前只是直接依赖候选；GPU-02 成功后必须保存完整 `pip freeze --all` 和 conda export，再由本地审计后升级为最终 lock。
3. `[PAPER]` 使用 BF16/DDP；`[CODE]` trainer 尚未实现真实 DDP，且 autocast dtype 路径不明确。环境可用不代表训练实现已经符合论文。
4. `[MISSING]` 论文未给训练 GPU 数量。8×A100-80GB 是后续全量训练的资源候选，不是论文事实。
5. `[CODE]` 官方仓库同时给出 `transformers>=4.47.0` 与 `peft>=0.18.0`，而 base-model card 指定的 Transformers preview commit 缺少 PEFT 0.18.0 导入的 `transformers.modeling_layers`。首次 CPU-02 实测复现了该冲突。`[INFERRED]` 采用 PEFT 0.18.0 官方示例使用的 Transformers 4.52.4；它仍满足仓库下界并包含 Qwen2.5-Omni。此差异必须在最终报告中保留。

## 4. CPU-02 安装与 GPU-02 验证范围

共享用户目录允许 CPU/GPU 容器读取同一个 Conda 环境，因此环境工作拆成：

- CPU-02：创建或恢复 `oea-repro`，安装 PyTorch cu126 与核心依赖，检查 imports、音频解码/重采样，并保存解析后的依赖和日志；
- GPU-02：只在 4090/A100 容器中运行 CUDA、BF16、NCCL 和最小矩阵运算验证，不再安装依赖。

默认下载源为阿里云 PyTorch cu126 wheel 目录和清华 PyPI 镜像；版本仍由本项目固定。阿里云页面是 wheel 列表而不是 PEP 503 simple index，因此脚本通过 `--find-links` 使用它，普通依赖通过清华 `--index-url` 解析。可分别通过 `PYTORCH_WHEELHOUSE_URL`、`PYPI_INDEX_URL` 临时覆盖，脚本会把实际 URL 写入日志。镜像选择标记为 `[INFERRED]`，不属于论文或官方代码配置。

- 阿里云索引：<https://mirrors.aliyun.com/pytorch-wheels/cu126/>
- 清华 PyPI 使用说明：<https://mirrors.tuna.tsinghua.edu.cn/help/pypi/>

全新安装使用 `scripts/setup_environment.sh`；之前安装被中断且环境已经存在时，使用 `scripts/resume_environment.sh`。恢复前必须保证没有任何其他服务器正在写共享的 `oea-repro`。两个脚本均不下载模型或数据、不安装 FlashAttention、不运行训练。

CPU-02 实测依赖安装和 `pip check` 已通过；随后发现 CPU 实例上的 `nvidia-smi` 会以 `Exec format error` 失败。环境检查器现将此类 `OSError` 记录为 `UNAVAILABLE`，CPU 模式不会因不可用的 GPU 管理命令崩溃；严格 CUDA/BF16/NCCL 判定仍留在 GPU 模式。

GPU-02 使用 `bash scripts/validate_gpu_environment.sh`，自动保存 commit、Git 状态、GPU 信息、完整日志和严格模式 JSON；不下载模型或数据。

DATA-02 预检确认共享环境与服务器均没有 `7zz`、`7z` 或 `7za`。数据工具通过 `scripts/install_data_tools.sh` 单独安装，默认使用清华 conda-forge 镜像并固定 `7zip=26.02`。该步骤只修改共享的 `oea-repro` Conda 环境，不使用 GPU，也不解压或修改数据；安装前会拒绝与另一个 Conda/Pip 安装并发执行，并保存安装前后包清单、explicit lock、resolved environment 和退出码。

## 5. Conda 环境检查误报修复

2026-07-16 的 MODEL-02 重启在下载前错误报告 `oea-repro` 不存在；原始失败目录为 `logs/model02_download_20260716_164527`。远程证据确认该环境处于 active 状态、环境目录和 Python 均存在，并且 `conda run -n oea-repro` 能加载 Torch 2.7.1。根因是脚本启用 `set -o pipefail` 后，使用 `grep -q` 检查 `conda env list`：匹配成功时 `grep` 提前退出，使上游进程收到 SIGPIPE，管道被误判失败。最小修复是去掉 quiet 模式并将完整匹配输出重定向到 `/dev/null`；同类检查在六个脚本中统一修复。该补丁只影响启动前环境存在性判断，不改变模型、数据、训练参数或实验结果。

随后 MODEL-04 在 tmux 子 shell 中触发了第二类环境解析问题。交互 shell 的 `conda` function 指向 `/home/jg525/miniconda3`，但其 `PATH` 中可执行文件解析为 `/opt/conda/bin/conda`；未导出的 function 不会自动进入 `bash scripts/...` 子进程，wrapper 因而从错误的 Conda base 激活环境。统一修复后的 wrapper 优先使用 Conda 导出的 `CONDA_EXE`，其次根据 `CONDA_PREFIX` 推导 base，最后才回退到 `PATH`。该补丁同样只影响环境定位，不修改任何实验配置或资源内容。
