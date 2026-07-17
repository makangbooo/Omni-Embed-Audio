# Omni-Embed-Audio 可审计复现指南

本文件是持续更新的复现入口，不是“已经全部复现”的声明。当前真实进度、阻塞项和下一步以
[`docs/reproduction_status.md`](docs/reproduction_status.md) 为准；论文全部实验清单见
[`docs/paper_experiment_inventory.md`](docs/paper_experiment_inventory.md)。论文与代码均未公开的配置始终标为
`[MISSING]`，根据实现作出的选择标为 `[INFERRED]`。

## 基本原则

- 分支固定为 `repro/oea-full`，正式运行前必须保存 `git rev-parse HEAD` 和 `git status --short`。
- CPU 数据处理、GPU embedding、CPU 指标汇总分开执行，适配项目目录在不同服务器之间挂载的工作方式。
- 模型、数据、checkpoint、embedding 和大型日志不提交到普通 Git；小型锁、配置、指标和报告可以提交。
- 任何预计超过 30 分钟的操作，都必须先单独汇报命令、GPU、显存、时间、磁盘和输出目录。
- 统一入口默认只打印计划。只有明确加入 `--execute` 才执行一个叶子阶段；长阶段还必须在资源报告获确认后加入
  `--acknowledge-long-operation`。
- `BLOCKED` 阶段不能通过强制参数绕过。应先补齐证据、更新状态和提交代码。

## 环境

远程统一使用：

```bash
source /home/jg525/miniconda3/etc/profile.d/conda.sh
conda activate /home/jg525/miniconda3/envs/oea-repro
cd /home/jg525/Omni-Embed-Audio
```

环境候选、已解析依赖和 GPU 验证记录见
[`docs/environment_notes.md`](docs/environment_notes.md)。不要使用 `base` 环境执行正式实验。

## 统一入口

查看全部注册阶段：

```bash
bash scripts/run_reproduction.sh --list
```

查看机器可读计划：

```bash
bash scripts/run_reproduction.sh --stage all --json
```

用户要求的公共入口已经注册：

```bash
bash scripts/run_reproduction.sh --stage official_eval
bash scripts/run_reproduction.sh --stage train_qwen3b
bash scripts/run_reproduction.sh --stage uiq_eval
bash scripts/run_reproduction.sh --stage baselines
bash scripts/run_reproduction.sh --stage all
```

这些组合阶段跨 CPU/GPU 服务器，因而只展开有序计划，不会在一台机器上擅自连续运行。真正执行时选择对应叶子阶段。

短 CPU 阶段示例：

```bash
bash scripts/run_reproduction.sh --stage data04 --execute
```

长 CPU 模型锁阶段示例（只有在事先完成资源汇报后）：

```bash
bash scripts/run_reproduction.sh \
  --stage official_model_lock \
  --variant oea_qwen3b_cl \
  --verify-existing-derived \
  --acknowledge-long-operation \
  --execute
```

CPU 指标阶段需要显式提供上一台 GPU 服务器生成的目录：

```bash
bash scripts/run_reproduction.sh \
  --stage official_eval_metrics \
  --embedding-dir /absolute/path/to/completed_embedding_run \
  --execute
```

阶段注册表固定在
[`configs/reproduction/stages.json`](configs/reproduction/stages.json)。注册表只允许项目已实现的非覆盖 wrapper；不会调用
`rm -rf`、`git reset --hard`、`git clean -fd` 或强制推送。

## 当前 CPU/GPU 边界

| 阶段 | 资源 | 当前说明 |
|---|---|---|
| DATA-04/05/06/07/08/09/10/11 | CPU | 下载固定小型资源、安全解压、manifest/泄漏/UIQ 校验 |
| 单变体资源审计、checkpoint 复核、OEA/vanilla 模型锁 | CPU | 读取量较大，真实运行前仍需长任务报告；vanilla 锁不加载 LoRA 或 projection |
| Caption/UIQ embedding | 1×A100-80GB | 严格离线、单卡、可恢复；必须使用已提交模型锁 |
| T2A/T2T/UIQ 指标 | CPU | 从固定 embedding 运行完整排名和指标 |
| Qwen3B 训练 | BLOCKED | 缺 world size、seed、精确 AudioCaps manifest、完整 stage 配置和 Clotho early-stop split |
| 基线 | BLOCKED | 7 个非 OEA 模型中仅 2 个具备完整静态代码入口、3 个具备固定资源身份、0 个可正式评测 |

基线逐模型代码入口、外部目录假设和资源身份审计见
[`docs/baseline_readiness.md`](docs/baseline_readiness.md)。该审计只读取源文件和本地文件元数据，不会安装依赖或下载权重。

## 证据链

正式 OEA 评测按以下顺序建立证据：

1. 固定 revision 的 base/checkpoint 资源审计；
2. 原始 checkpoint 结构审计；
3. inference-only 权重安全提取或已有权重逐张量只读复核；
4. 生成并审阅 `results/model_locks/<variant>.json`；
5. 将小型模型锁提交并在 GPU 服务器拉取；
6. 由已提交协议和模型锁生成不可变 resolved config；
7. 生成 embedding；
8. 在 CPU 服务器重新核对协议、模型锁、Git、输入/输出哈希后计算指标。

详细契约见
[`docs/official_checkpoint_preparation.md`](docs/official_checkpoint_preparation.md) 和
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)。

## 尚未完成

当前仍不能宣称论文完整复现。至少还缺：AudioCaps 与 MECAT 正式音频候选集、六个 OEA 变体完整官方评测、四个基线、首个
3B 训练闭环、其余模型训练、负查询 target/HN 官方配对、效率实验和最终逐表差异报告。失败与缺失项不会从最终报告中删除。
