# Omni-Embed-Audio 复现状态

最后更新：2026-07-16（Asia/Shanghai）

状态值仅使用：`TODO`、`IN_PROGRESS`、`WAITING_USER`、`RUNNING_REMOTE`、`COMPLETED`、`FAILED`、`BLOCKED`。

| 阶段 | 任务 | 状态 | 当前 Commit | 远程状态 | 阻塞原因 | 下一步 |
|---|---|---|---|---|---|---|
| 0 项目审计 | 完整阅读论文正文/附录/表/图/脚注 | COMPLETED | `48d7a50` | fork 已同步 | 无 | 固定 inventory |
| 0 项目审计 | 审计 README、依赖、训练/评测、数据、UIQ、HN、指标 | COMPLETED | `48d7a50` | fork 已同步 | 无 | 用户确认审计结论 |
| 0 项目审计 | 31 条实验/图/派生分析 inventory | COMPLETED | `48d7a50` | fork 已同步 | 无 | 用户确认复现边界 |
| 0 GitHub | 建立 `repro/oea-full` 分支 | COMPLETED | `48d7a50` | 本地与 fork 分支均存在 | 无 | 后续提交仅推送该分支 |
| 0 GitHub | commit 并 push 第一阶段审计 | COMPLETED | `48d7a502e279bd9a2a3195352163626fb1781a3b` | `origin/repro/oea-full` 已建立 | 无 | 保持本地与远程实验 commit 一致 |
| 1 环境 | CPU-01：CPU/内存/存储/编译器/网络审计 | COMPLETED | `20770d0` | 远程完成；原始日志未进 Git | Zenodo 返回 503；容器 `nproc=2` | 在共享目录挂载到 A100 后执行 GPU-01 |
| 1 环境 | GPU-01：A100/Driver/CUDA/NCCL 基础审计 | COMPLETED | `905e84d` | 1×A100-SXM4-80GB 远程完成；原始日志未进 Git | 多卡 GPU–GPU 拓扑留待 DDP smoke | 固定环境候选并执行 GPU-02 |
| 1 环境 | 生成环境候选、安装脚本与无模型检查 | COMPLETED | `c61fecb` | fork 已同步 | 完整 transitive lock 必须在 Linux solve 后生成 | 固定首批模型资源 |
| 1 环境 | CPU-02：创建 `oea-repro` 并生成 resolved lock | COMPLETED | `11f20d2` | `pip check`、必需接口、仓库导入、音频 I/O/重采样均通过；resolved artifacts 已生成 | 无 | 在单卡实例执行 GPU-02 |
| 1 环境 | GPU-02：CUDA/BF16/NCCL 验证 | COMPLETED | `711ef28` | 1×A100-SXM4-80GB 验证通过；原始日志保存在共享目录 | 无；`flash-attn` 为非必需可选项 | 下载首批固定 revision 模型资源 |
| 2 官方权重 | MODEL-01：Qwen2.5-Omni-3B + OEA-Qwen3B-Cl | COMPLETED | `bba6084` | 断点续传完成；两个 asset 和全部 LFS SHA256 通过；无临时文件/下载进程 | 无；首次失败证据保留在旧运行目录 | 审计 checkpoint 内部结构 |
| 2 官方权重 | MODEL-02：OEA-Qwen3B-AC checkpoint | WAITING_USER | `30f003a` | 等待 CPU 服务器并行下载 | 尚未下载固定 revision 的 `step_350.pt` | 与 DATA-01 分开启动，返回下载 manifest 和 LFS SHA256 |
| 2 官方权重 | 5 个 Clotho 样例 Qwen3B-Cl smoke | COMPLETED | `fba8c3c` | 运行 `..._20260716_130619` 严格离线成功；5 音频/5 查询、544 LoRA、双 head、512 维归一化 embedding 全部通过；峰值 9.141 GiB allocated | 无；首次失败 `..._20260716_125636` 仍保留 | 固定小型结果摘要并进入检索指标单元测试 |
| 2 官方权重 | Tables 2/3：单个 3B 官方权重 T2A/T2T | BLOCKED | N/A | 未开始 | 数据未准备；官方 OEA eval 命令不加载 checkpoint | 先补 canonical evaluator 与 manifest |
| 2 指标 | Figure 3 / Table 17 指标单元测试 | COMPLETED | `88a7d1f` | R@k、Δ-Rank、HNSR、HNSR@k、TFR、TFR-HN@k 已按论文公式实现；7 个合成测试及完整 24 项测试通过 | 无；正式表 17 仍缺 target-HN audio ID | 在 canonical negative evaluator 中接入已验证指标 |
| 2 UIQ | Tables 12–15 正向 UIQ | BLOCKED | N/A | 未开始 | loader schema 不匹配；音频未准备 | 加 schema adapter 与数据校验 |
| 2 Negative | Tables 4/17 否定查询 | BLOCKED | N/A | 未开始 | 发布文件无 HN audio ID | 请求作者 pairing 或审计式重建 |
| 3 数据 | WavCaps ≤31s + leakage blocklists | BLOCKED | N/A | 未开始 | 精确 manifest/blocklists 未发布 | 先做 metadata-only overlap audit |
| 3 数据 | AudioCaps v2 91,256 manifest | BLOCKED | N/A | 未开始 | 数据版本/下载源未公开 | 请求作者或获得用户已有文件 |
| 3 数据 | Clotho v2.1 evaluation 下载与 checksum | COMPLETED | `aff03e4` | DATA-01 首次下载及两次幂等复核完成；3 个 MD5 全部匹配 | 无；两次复核均为 `verified_existing`，未重复下载 | DATA-02 安全解压、1,045 条 WAV/CSV/UIQ ID 校验 |
| 3 数据 | MECAT 847/848 manifest | BLOCKED | N/A | 未开始 | 论文与 UIQ release 数量不一致 | 固定官方评测子集 |
| 4 训练 | 首个 3B 过拟合/单卡/DDP/全量闭环 | BLOCKED | N/A | 未开始 | DDP、seed、早停、stage LR 等不完整 | 评测闭环后再补最小训练基础设施 |
| 5 基线 | LAION/Robust/MGA/M2D-CLAP | BLOCKED | N/A | 未开始 | checkpoint revision/统一入口不完整 | 官方 OEA 评测完成后逐个固定资源 |
| 6 扩展 | 人工/闭源 API、效率、token-length、派生图表 | TODO | N/A | 未开始 | 按阶段后置 | 核心训练评测完成后执行 |

## 当前焦点

- 当前阶段：环境、首批官方权重、checkpoint 结构审计、inference-only 权重提取及单卡 GPU smoke 已完成。
- 当前对应论文范围：所有主表 1–5、附录表 6–17、图 1–3、附录 A–M 已建清单。
- 最近完成：在 A100-SXM4-80GB 上确认 PyTorch CUDA 12.6、BF16、NCCL 2.26.2、必需 Qwen/PEFT 接口、仓库导入和音频 I/O 均通过；固定首批模型的不可变 Hugging Face revision。
- 当前阻塞：正式论文表格评测仍缺完整 AudioCaps、Clotho 和 MECAT/UIQ 数据；当前 5 条样例只证明首个官方 checkpoint 的最小前向闭环，不代表论文 Recall 复现。
- 下一步：DATA-01 已完成；准备 DATA-02 安全解压、1,045 条 WAV 解码与 captions/metadata/UIQ ID 对齐，同时等待 MODEL-02 下载结果。
