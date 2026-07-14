# Omni-Embed-Audio 复现状态

最后更新：2026-07-15（Asia/Shanghai）

状态值仅使用：`TODO`、`IN_PROGRESS`、`WAITING_USER`、`RUNNING_REMOTE`、`COMPLETED`、`FAILED`、`BLOCKED`。

| 阶段 | 任务 | 状态 | 当前 Commit | 远程状态 | 阻塞原因 | 下一步 |
|---|---|---|---|---|---|---|
| 0 项目审计 | 完整阅读论文正文/附录/表/图/脚注 | COMPLETED | `48d7a50` | fork 已同步 | 无 | 固定 inventory |
| 0 项目审计 | 审计 README、依赖、训练/评测、数据、UIQ、HN、指标 | COMPLETED | `48d7a50` | fork 已同步 | 无 | 用户确认审计结论 |
| 0 项目审计 | 31 条实验/图/派生分析 inventory | COMPLETED | `48d7a50` | fork 已同步 | 无 | 用户确认复现边界 |
| 0 GitHub | 建立 `repro/oea-full` 分支 | COMPLETED | `48d7a50` | 本地与 fork 分支均存在 | 无 | 后续提交仅推送该分支 |
| 0 GitHub | commit 并 push 第一阶段审计 | COMPLETED | `48d7a502e279bd9a2a3195352163626fb1781a3b` | `origin/repro/oea-full` 已建立 | 无 | 保持本地与远程实验 commit 一致 |
| 1 环境 | 远程 A100 系统信息与网络检查 | WAITING_USER | N/A | 未开始 | 尚未获得具体服务器输出/显存规格 | 选择一台 A100，运行只读检查 |
| 1 环境 | 创建 `oea-repro` 与 lock 文件 | TODO | N/A | 未开始 | 等待系统信息 | 核对 Driver 后锁 Python/PyTorch/CUDA/transformers |
| 2 官方权重 | 5 个 Clotho 样例 Qwen3B-Cl smoke | TODO | N/A | 未开始 | 环境未完成 | 修正实际 checkpoint 文件名并前向 |
| 2 官方权重 | Tables 2/3：单个 3B 官方权重 T2A/T2T | BLOCKED | N/A | 未开始 | 数据未准备；官方 OEA eval 命令不加载 checkpoint | 先补 canonical evaluator 与 manifest |
| 2 指标 | Figure 3 / Table 17 指标单元测试 | TODO | N/A | 未开始 | 无 | 实现 HNSR/TFR/Δ-Rank 并用合成例验证 |
| 2 UIQ | Tables 12–15 正向 UIQ | BLOCKED | N/A | 未开始 | loader schema 不匹配；音频未准备 | 加 schema adapter 与数据校验 |
| 2 Negative | Tables 4/17 否定查询 | BLOCKED | N/A | 未开始 | 发布文件无 HN audio ID | 请求作者 pairing 或审计式重建 |
| 3 数据 | WavCaps ≤31s + leakage blocklists | BLOCKED | N/A | 未开始 | 精确 manifest/blocklists 未发布 | 先做 metadata-only overlap audit |
| 3 数据 | AudioCaps v2 91,256 manifest | BLOCKED | N/A | 未开始 | 数据版本/下载源未公开 | 请求作者或获得用户已有文件 |
| 3 数据 | Clotho v2.1 manifests | TODO | N/A | 未开始 | 尚未下载 | 环境后下载并校验 Zenodo checksum |
| 3 数据 | MECAT 847/848 manifest | BLOCKED | N/A | 未开始 | 论文与 UIQ release 数量不一致 | 固定官方评测子集 |
| 4 训练 | 首个 3B 过拟合/单卡/DDP/全量闭环 | BLOCKED | N/A | 未开始 | DDP、seed、早停、stage LR 等不完整 | 评测闭环后再补最小训练基础设施 |
| 5 基线 | LAION/Robust/MGA/M2D-CLAP | BLOCKED | N/A | 未开始 | checkpoint revision/统一入口不完整 | 官方 OEA 评测完成后逐个固定资源 |
| 6 扩展 | 人工/闭源 API、效率、token-length、派生图表 | TODO | N/A | 未开始 | 按阶段后置 | 核心训练评测完成后执行 |

## 当前焦点

- 当前阶段：项目审计与 GitHub 发布测试完成，等待远程 A100 系统信息。
- 当前对应论文范围：所有主表 1–5、附录表 6–17、图 1–3、附录 A–M 已建清单。
- 最近完成：论文/代码参数冲突审计、UIQ release 完整性统计、六个 HF checkpoint 实际文件/revision/大小核验。
- 当前阻塞：尚无远程系统信息；关键数据/HN pairing 未提供。
- 下一步：用户确认第一阶段审计后，先做环境检查与 5 样例官方权重 smoke，不下载数据集或启动训练。
