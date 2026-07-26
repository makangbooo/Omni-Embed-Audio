# ASR-Uncertainty-Guided Reranking over OEA：实验计划

版本：Phase 0 protocol locked，2026-07-26

## 1. 研究范围

一级召回固定为 OEA 音频 query 对冻结文本 corpus 的 Top-100。二级模块仅重排
这 100 个候选，不扩充候选集合。核心比较是：

```text
audio
  ├─ OEA audio encoder ────────────────> OEA Top-100 + s_oea
  └─ Whisper 1/4-best + uncertainty ──> BGE reranker scores
                                              │
                    OEA/ASR score + rank disagreement
                                              │
                         candidate-level gate g(q, d)
                                              │
                                   final reranked Top-100
```

第一阶段冻结 OEA、Whisper、BGE dense retriever 和 BGE reranker，只训练小型
candidate-level gate。任何 OEA/Whisper/reranker 微调都不在当前授权范围。

### 1.1 当前主实验与非主实验

当前主实验是 FiQA 四种 SQuTR 声学条件上的 `B1`–`B7`、query-level gate 与
candidate-level uncertainty gate 对照，主指标为 nDCG@10；NQ 只在 FiQA
Go/No-Go 和用户 GPU 批准后做跨领域零样本验证。OEA-Nemo3B (+Cl) 的 Clotho
复现、SQuTR 内容审计和 FiQA 一级召回 Go/No-Go 都是主实验的前置门禁或基线，
不是本项目最终创新方法本身。

暂停的 PathCell 实验、OEA 全论文复现以及与候选级动态门控无关的新数据集/模型
不属于当前范围。已有 OEA 结果只作为 checkpoint/protocol/效率证据冻结复用，不
计作本项目方法结果。

## 2. 数据与数据泄漏约束

| 数据 | 用途 | 允许的监督 |
|---|---|---|
| FiQA train（5,500 query，待文件验证） | gate 训练 | 仅 train qrels |
| FiQA dev（500 query，待文件验证） | 特征、归一化、固定融合与 gate 超参 | 仅 dev qrels |
| SQuTR FiQA test（648 query × 4 条件） | 正式测试 | 禁止训练/选择超参 |
| SQuTR NQ test（3,452 query × 4 条件） | 跨领域 zero-shot | 禁止训练/选择超参 |

SQuTR 官方 Hugging Face 发布页标注许可证为 `CC BY-SA 4.0`，公开下载不收取
数据费用；使用、改编和发布派生物时仍须遵守署名与相同方式共享条款。当前归档已
包含 `en/fiqa` 与 `en/nq` 的 corpus、queries、qrels 和四条件音频，不重复下载。

强制检查：

1. query ID 精确交集；
2. Unicode/空白/大小写规范化后的 query text 交集；
3. TTS 音频 ID 到源 query ID 的一一映射；
4. train/dev/test speaker 集互斥（若 metadata 足以验证）；
5. train/dev/test noise 文件或时间片互斥；
6. qrels 只由相应 split 加载；
7. NQ 结果产生后不再修改 FiQA 选择的 gate/归一化/融合超参。

Gold transcript 只用于 U1/U2 上限和 WER，不进入正式推理特征。

## 3. 待锁定的模型协议

| 组件 | 固定候选 | revision | 初始协议 |
|---|---|---|---|
| OEA | OEA-Nemo3B (+Cl) | checkpoint repo `958891...`; base `865db1...` | LoRA + 双 512-d heads、mean pool、L2 |
| 原始 Omni | `nvidia/omni-embed-nemotron-3b` | `865db1...` | hidden mean pool、L2；与 OEA 分栏 |
| ASR | `openai/whisper-large-v3` | `06f233...` | 16 kHz、English transcribe、timestamps off |
| dense | `BAAI/bge-base-en-v1.5` | `a5beb1...` | CLS pooling、768-d、L2、max length 512 |
| reranker | `BAAI/bge-reranker-v2-m3` | `953dc6...` | frozen sequence classification score |

### 3.1 OEA protocol decision

用户于 2026-07-26 确认采用发布代码协议：文本候选使用 `query:`，audio-only
message 不人为加入 `passage:` 文本元素。发布 checkpoint 与公开 adapter 可直接
审计；论文文字协议另列为 `[PAPER-CONFLICT]`。不根据测试结果更换主协议。

### 3.2 文档构造

按照 SQuTR official loader：

```text
title + "\n" + text      if title is non-empty
text                     otherwise
```

不做额外 HTML 清理、句子截断或关键词抽取。超长文档只按模型 tokenizer 的固定
`max_length` 截断，截断统计写入 manifest。

### 3.3 BGE query instruction

在看到 FiQA test 结果前锁定。建议用模型卡的 English short-query retrieval
instruction：

```text
Represent this sentence for searching relevant passages:
```

文档不加 instruction。必须在 FiQA dev 上与“无 instruction”至多做一次预注册
选择；如果用户希望完全对齐 SQuTR BGE runner，则固定其实际 prompt，不再选择。

### 3.4 Whisper 4-best 与“后验”

计划使用 Transformers beam search：

- `num_beams=4`
- `num_return_sequences=4`
- `do_sample=false`
- `language=en`
- `task=transcribe`
- `return_dict_in_generate=true`
- `output_scores=true`

保存 raw sequence score、逐 token transition score、有效 token 数和平均 token
log-prob。若接口不提供严格概率，定义

```text
proxy_logit_m = mean_valid_token_logprob_m
p_m = softmax(proxy_logit_m / T_asr)
```

并始终命名为 `proxy posterior`，不宣称声学模型严格后验。`T_asr` 只能在 FiQA
dev 选择，默认和候选集合一起写入配置。额外不确定性包括：

- 归一化 N-best entropy；
- Top1/Top2 proxy probability margin；
- 4 个假设两两 normalized edit distance 的均值/最大值；
- top-1 average token log-prob；
- Whisper no-speech probability，仅在 API 可稳定获得时启用并做缺失标志。

## 4. 基线与诊断矩阵

| ID | 方法 | 候选集合 | 训练 |
|---|---|---|---|
| B1 | Whisper 1-best + BGE dense | 全 corpus | 无 |
| B2 | 原始 Omni-Embed-Nemotron audio→text | 全 corpus | 无 |
| B3 | OEA-Nemo3B (+Cl) audio→text | 全 corpus | 无 |
| B4 | OEA Top-100 + 1-best Cross-Encoder | OEA Top-100 | 无 |
| B5 | OEA Top-100 + 固定权重融合 | OEA Top-100 | FiQA dev 选固定权重 |
| B6 | OEA Top-100 + RRF | OEA Top-100 | FiQA dev 选固定 `k` 或预注册 |
| B7a | 4-best CE 等权平均 | OEA Top-100 | 无 |
| B7b | 4-best CE 最大值 | OEA Top-100 | 无 |
| B7c | 4-best proxy-posterior 聚合 | OEA Top-100 | FiQA dev 选温度 |
| QG | query-level gate | OEA Top-100 | FiQA train/dev |
| Ours | candidate-level uncertainty gate | OEA Top-100 | FiQA train/dev |
| U1 | Gold query + BGE dense | 全 corpus | 无 |
| U2 | Gold query + CE rerank | OEA Top-100 | 无 |
| U3 | OEA Top-100 oracle | OEA Top-100 | 无 |
| U4 | OEA candidate Recall@20/50/100 | OEA Top-100 | 无 |

所有重排方法复用同一 OEA Top-100 文件，禁止因方法不同重新生成候选。

## 5. 分数与门控定义

### 5.1 可比较归一化

每个 query 的同一 Top-100 内分别计算 OEA 和 ASR 分数的 z-score：

```text
z(s_i) = (s_i - mean(s)) / max(std(s), epsilon)
```

并同时实现 deterministic rank normalization 作为消融。默认方法必须在 FiQA dev
前固定；不得直接相加原始 cosine 和 CE logits。

### 5.2 N-best 聚合

```text
s_asr(q,d) = logsumexp_m(log p_m + r(m,d) / T_ce)
```

`T_ce` 默认 1.0；若调参，只能使用 FiQA dev。等权、max 和 1-best 使用同一
已缓存 `r(m,d)`，避免重复 GPU 计算。

### 5.3 Candidate-level gate

候选特征最小集合：

- ASR top-1 average token log-prob；
- normalized N-best entropy；
- Top1/Top2 margin；
- hypothesis edit-distance mean/max；
- normalized OEA/ASR score；
- OEA/ASR rank 与 rank gap；
- 两路 Top-10/20/50 overlap（query-level，广播到候选）；
- 缺失/退化特征标志。

```text
g(q,d) = sigmoid(MLP(features(q,d)))
S(q,d) = (1-g) * z_oea(q,d) + g * z_asr(q,d)
```

训练候选来自 OEA Top-100。每个 query group 采约 16 个候选，保留全部正例或至少
一个正例，其余从 OEA 高排位非相关文档中确定性采样。初始损失为多正例 listwise
softmax；不叠加未经消融的辅助 loss。

## 6. 阶段计划与门禁

### Phase 0：审计和协议草案

- 状态：本地完成；等待用户确认。
- 产物：本计划、workspace audit、STATUS、ARTIFACTS。
- 未执行：下载、GPU、训练。

### Phase 1：OEA 正确性检查

1. CPU 只读实时复核 Nemo base/+Cl snapshot；
2. 生成 inference-only checkpoint 和 immutable model lock；
3. 五音频/五文本 smoke，检查 512 维、finite、L2 norm；
4. Clotho 1,045 audio / 5,225 captions 全量 T2A；
5. 复现表 2 OEA-Nemo3B (+Cl)：
   R@1/5/10 = `21.57/47.16/60.36`。

远程 CPU 节点首先运行 `scripts/run_asrur_nemo_phase1_audit.sh`。该 runner 将完整
资源审计、checkpoint 准备和 portable model lock 写入被 Git 忽略的独立
`logs/asrur_nemo_phase1_audit_<timestamp>/`；本地 Codex 验证返回证据后，才将
小型 canonical lock 提交到 `results/model_locks/`。这样远程不会产生已跟踪修改，
也不会出现本地与远程分别手改模型身份文件的情况。

超过任一指标 2 个百分点时停止后续阶段，审计 checkpoint、prefix、pooling、
normalization、方向和 caption positives。

### Phase 2：FiQA 一级召回 Go/No-Go

前置：DATA-13B complete；FiQA corpus/qrels schema 和文本构造锁定。

执行 B1/B2/B3 和 U4，四种声学条件全部 648 query，不抽 test 子集。保存文档
embedding、query embedding、Top-100、qrels 和逐 query 指标。

Go 条件：

- OEA Recall@100 最好不低于 80%；
- Top-100 oracle nDCG@10 相比裸 OEA 至少高 0.08；
- OEA Top-100 中存在可观精排空间；
- OEA 没有弱于原始 Omni 到失去合理基线资格。

不满足时报告并等待决定；不擅自改为双路召回。

### Phase 3：冻结普通精排基线

在唯一 OEA Top-100 上计算并缓存 B4/B5/B6、U1/U2。所有固定融合参数只用 FiQA
dev，正式 test runner 读取 frozen config。

### Phase 4：4-best 与门控

1. 实现 Whisper 4-best 与 score 校验；
2. 实现 1-best/equal/max/proxy-posterior 聚合；
3. 实现 query-level 与 candidate-level gate；
4. 在用户批准的 FiQA train/dev TTS 上训练；
5. 仅用 dev early stopping 和选参。

TTS 是独立审批点。建议优先研究与 SQuTR 同族的 CosyVoice-3，但说话人参考、
许可证、train/dev speaker 隔离和 DEMAND/NOISEX-92 切分必须形成可执行方案后再
下载/合成。当前不把某个 TTS 选择写成已决定事实。

### Phase 5：FiQA 正式评测

- Clean/20/10/0 dB；
- 三个 gate seed；
- A1–A9 必要消融；
- paired bootstrap；
- WER 分桶、gate 分布和失败案例；
- 主指标 nDCG@10；同时报告 MRR@10、Recall@10/20/50/100、WER 和噪声下降。

### Phase 6：NQ zero-shot

只有 FiQA 达到用户给定六项条件并取得新的 GPU 批准后执行。NQ 不训练、不选参，
只运行必要主方法和上限，不重复全部消融。

### Phase 7：统计、效率与报告

生成 CSV、Markdown/LaTeX、bootstrap、曲线、参数量、显存、吞吐、延迟、失败
案例和最终报告。明确区分已验证、推断、失败、未完成、泄漏风险和创新性边界。

## 7. 实现模块

计划使用独立 Python package `AudioRetrieval/asr_uncertainty_reranking/`：

| 模块 | 责任 |
|---|---|
| `schema.py` | corpus/query/qrels/N-best/cache schema |
| `cache_manifest.py` | hash、模型 revision、协议相容性与拒绝静默复用 |
| `data.py` | FiQA/NQ/SQuTR loader 与 split leakage audit |
| `oea_encoder.py` | OEA/vanilla Omni 统一、但结果分栏的编码接口 |
| `bge_dense.py` | BGE document/query encoding 和分块检索 |
| `whisper_nbest.py` | 1/4-best、token/sequence score、不确定性 |
| `cross_encoder.py` | Top-100 × N-best frozen CE 评分 |
| `aggregation.py` | 1-best/equal/max/proxy-posterior |
| `normalization.py` | query-local z/rank normalization |
| `features.py` | 候选级和 query 级特征 |
| `gate.py` | query/candidate gate 与 listwise loss |
| `metrics.py` | nDCG/MRR/Recall/Oracle/WER/noise degradation |
| `bootstrap.py` | paired bootstrap |
| `plotting.py` | 噪声、WER、gate 图 |

每个长任务先生成 immutable config 和 cache manifest，支持 resume、失败样本表、
结构化日志、原子写和不覆盖已有 final artifact。

### 7.1 已完成的无模型 CPU 核心

commit `8cc5984dc35a272934d434993006814855f39624` 已实现：

- `normalization.py`：query-local population z-score、平均秩 normalization、确定性 tie-break；
- `aggregation.py`：Whisper proxy posterior、归一化熵、Top1/Top2 margin、N-best
  编辑距离、one/equal/max/proxy-posterior logsumexp；
- `metrics.py`：graded nDCG、MRR、multi-positive Recall、固定候选 Oracle、clean-to-noise
  下降和 micro WER；
- `cache_manifest.py`：严格 JSON schema、SHA256、兼容性字段级差异、默认 Git commit
  身份检查和不覆盖式原子发布。

相关 38 项 CPU 测试通过。该完成状态只证明公式和缓存契约已验证，不代表 Whisper、
BGE、OEA 模型推理或任何实验结果已完成。

### 7.2 已完成的主实验 CPU 流水线

commit `8f28c5590810f3e430d3ce7591266ad7f8e38f21` 继续实现：

- FiQA corpus/query/qrels 与 SQuTR audio manifest 严格装配、count/closure 和
  train/dev/test ID/规范化文本泄漏审计；
- 缓存 embedding 的 exact chunked Top-K reference，以及 B1/B2/B3/U1 全 corpus
  ranking 评测；
- OEA Top-100、Whisper 4-best、Cross-Encoder score 的严格 query/candidate
  identity 与顺序绑定；
- FiQA-dev 固定融合、RRF 与两个温度的预注册网格选择；
- query-level/candidate-level NumPy MLP gate、三随机种子、multi-positive
  listwise loss、zero-positive-candidate query 显式排除记录；
- B4–B7、U2–U4、A1–A9、paired bootstrap、clean-to-0dB 降幅、gate 分布与
  统一 CSV 汇总；
- non-overwrite 输出、输入 SHA256、Git commit/status 和命令行 provenance。

本地 58 项 ASRUR 专项测试和 345 项全仓测试通过。端到端测试只使用合成缓存，
其输出显式标为 `synthetic_smoke_only_not_a_research_result`；没有下载、模型推理、
GPU 计算、TTS 或真实 gate 训练。

## 8. 首批资源审批草案

以下仅列出，Phase 0 未下载。

| ID | 资源与固定 revision | 许可证 | 最小下载 | 预留磁盘 | 保存目录 | 必要性 | 小替代 |
|---|---|---|---:|---:|---|---|---|
| D1 | `mteb/fiqa@5e59eeb...` | `unknown` | 48,616,245 B | 0.1 GB | `/home/jg525/datasets/oea/fiqa_mteb` | train/dev gate 数据与泄漏检查 | 已排除重复 TSV 和 README；corpus/queries/三 split qrels JSONL 全部保留 |
| D2 | `openai/whisper-large-v3@06f233...` minimal safetensors | Apache-2.0 | 3,091,519,764 B | 3.5 GB | `/home/jg525/models/whisper-large-v3` | B1、1/4-best、不确定性、WER | `turbo` 更小但改变用户指定基线，不作为正式替代 |
| D3 | `BAAI/bge-base-en-v1.5@a5beb1...` minimal safetensors | MIT | 438,900,399 B | 0.6 GB | `/home/jg525/models/bge-base-en-v1.5` | B1/U1 dense | small 版更小但改变强基线 |
| D4 | `BAAI/bge-reranker-v2-m3@953dc6...` minimal safetensors | Apache-2.0 | 2,293,242,108 B | 2.6 GB | `/home/jg525/models/bge-reranker-v2-m3` | B4–B7 和 Ours | 无同协议小替代；可先用 mock 完成 CPU 测试 |

首批固定最小文件集合计 5,872,278,516 bytes（约 5.47 GiB），建议预留 7 GB。Nemo/SQuTR 不重复
下载；独立 NQ snapshot 和 TTS/noise 资源均延后审批。

用户已于 2026-07-26 批准 D1–D4。批准仅覆盖下载与内容校验，不覆盖 GPU
inference、TTS、微调或上传。

精确来源：

- `https://huggingface.co/datasets/mteb/fiqa`
- `https://huggingface.co/openai/whisper-large-v3`
- `https://huggingface.co/BAAI/bge-base-en-v1.5`
- `https://huggingface.co/BAAI/bge-reranker-v2-m3`

## 9. 首个 GPU 阶段草案

首个 GPU 阶段是 **G1：Phase 1 OEA-Nemo3B (+Cl) 正确性检查**：

- 模型：固定 Nemo base + `OEA-Nemo3B-Cl/step_450_best.pt`；
- 数据：已有 Clotho v2.1 evaluation；
- GPU：1×RTX 4090 24GB（用户于 2026-07-26 确认）；
- 初估峰值：12–18 GiB（正式前先用 5/25 smoke 实测）；
- 初估时长：smoke 5–15 分钟，全量 embedding 30–90 分钟；
- 磁盘：约 0.2 GB 新 embedding/日志，不覆盖已有结果；
- 可恢复性：逐批缓存；失败保留 attempt 和 failure manifest；
- 中断：不会修改原始模型/数据，临时文件与 final 分离。

4090 用于 checkpoint 正确性和检索指标；这些结果可与论文 Recall 对照，但效率、
绝对延迟和显存不得冒充论文 A100 行。精确 wrapper、commit、RUN_ID 和输出目录
将在 Nemo 实时 CPU 审计/model lock 完成并提交后给出；在那之前不启动 GPU。

## 10. 当前等待的决定

1. audio-only no-prefix 主协议已确认；
2. D1–D4 首批下载已批准；
3. G1 的 1×RTX 4090 24GB 资源计划已确认；精确命令提交前不启动 GPU；
4. DATA-13C attempt 3 因六子集全局锁申请失败退出；旧 lock 保留为失败证据。
   主实验仅需要 `en/fiqa` 和 `en/nq`，因此使用 DATA-13D 的独立输出、缓存和
   scoped lock 校验 16,400 条目标音频；这不会修改或接管旧 lock；
5. 用户已确认 Phase 0，D1–D4 下载可在独立 CPU 服务器与 DATA-13C 锁审计并行；
   G1 GPU 仍必须等待 DATA-13D、Nemo live audit、model lock 和精确命令；
6. `[INFERRED][USER-CONFIRMED 2026-07-26]` B5/B6 主融合固定使用 4-best
   proxy-posterior 分数，以便 A3 只比较融合策略；1-best 融合只保留为辅助
   诊断。该选择已在任何正式 FiQA dev/test 或 NQ 结果产生前锁定。
