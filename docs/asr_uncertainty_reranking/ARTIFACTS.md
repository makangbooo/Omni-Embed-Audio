# ASR-Uncertainty Reranking 资产登记

## ICASSP scope and real-model adapter milestone

| Artifact | Identity | Path | Status |
|---|---|---|---|
| ICASSP required-experiment checklist | 28 experiment packages; OEA-only baseline scope; no synthetic result counted | `docs/asr_uncertainty_reranking/ICASSP_EXPERIMENT_CHECKLIST.md` | COMPLETE at code commit `207bcd0` |
| Frozen local-model adapters | BGE dense CLS/L2, BGE v2-M3 CE, Whisper four-beam proxy score, explicit 16 kHz resampling | `AudioRetrieval/asr_uncertainty_reranking/model_adapters.py` | IMPLEMENTED_NOT_RUN; 371 full-repository tests passed; no model/GPU execution |
| Resumable real-cache producers | BGE dense/CE, Whisper 4-best, OEA-Nemo, original Omni, exact Top-100; per-condition FiQA-qrels ID contract | `scripts/generate_asrur_frozen_caches.py`, `scripts/generate_asrur_omni_caches.py`, `scripts/plan_asrur_phase2.py` | IMPLEMENTED_NOT_RUN at `dd44bc4`; 381 full-repository tests passed; no model/GPU execution |
| Portable vanilla-Nemotron lock audit | exact local base snapshot audit and untracked portable lock evidence | `scripts/run_asrur_vanilla_nemo_phase2_audit.sh`; compact audit `results/audits/asrur_vanilla_nemo_portable_lock_20260727.json` | COMPLETE at execution `7cfbfab`; pipeline/wrapper=`0/0`; portable and canonical SHA256 both `eb62da75...cc1c` |
| Nemo Clotho A2T direction suite | 1,045 audio queries against 5,225 frozen caption candidates; five matching captions are multi-positive qrels | `results/audits/asrur_nemo_clotho_a2t_20260727.json` | COMPLETE at execution `7cfbfab`; R@1/5/10=`26.8900/51.3876/65.2632`; CPU-only; not paper-reported |
| Phase-2 frozen-retrieval orchestrator | OEA/vanilla Omni/BGE/Whisper frozen inference, exact full-corpus rankings, dev-only BGE template selection, four-condition metrics and OEA candidate oracle | `scripts/run_asrur_phase2_frozen_retrieval.sh`, `scripts/run_asrur_phase2_attempt3.sh`, `scripts/select_asrur_bge_query_template.py` | WAITING_USER; teacher-forced proxy smoke passed; tmux-only attempt-3 entrypoint requires exactly 11 checksummed read-only OEA/vanilla/BGE cache reuses before generating four Whisper conditions and final rankings |
| Phase-2 Whisper dtype failure | Attempt-2 per-record failure evidence and exact systemic exception | `results/audits/asrur_phase2_whisper_dtype_failure_20260727.json` | FAILED at execution `ee63772`; 648/648 records had FP32 input versus BF16 conv bias; no completion manifest or formal metric; repair is `5835fab` |
| Phase-2 Whisper dtype repair smoke | Strict-offline, one-record, four-beam RTX-4090 gate with D2–D4 resource audit, finite proxy-score validation, hashes and elapsed time | `scripts/run_asrur_whisper_dtype_smoke.sh`, `tests/test_run_asrur_whisper_dtype_smoke.py`; failure audit `results/audits/asrur_whisper_beam_index_smoke_failure_20260727.json` | FAILED at execution `6ef3c2c`; dtype mismatch was fixed and D2–D4 passed, but Transformers 4.52.4 Whisper wrapper re-stacked scores without rebasing global beam indices, causing a CUDA gather assertion |
| Phase-2 Whisper teacher-forced proxy repair | Explicit English/transcribe/no-timestamps prompt, frozen base `GenerationMixin` four-beam path, and exact-sequence float32 teacher-forced conditional log-probability | `AudioRetrieval/asr_uncertainty_reranking/model_adapters.py`, `tests/test_asrur_model_adapters.py`; success audit `results/audits/asrur_whisper_teacher_forced_smoke_success_20260728.json` | COMPLETE at execution `54d0e72`; one-record RTX-4090 smoke produced exactly four finite proxy scores, kept finite beam sequence scores separately, used no transition scores, wrapper=`0`, elapsed=`48s` |
| Phase-2 attempt-3 single-record failure | Exact failure, record/audio metadata, shard counts and prohibited-repair policy | `results/audits/asrur_phase2_attempt3_single_record_failure_20260728.json` | FAILED at execution `bc4450a`; three Whisper conditions complete, snr_0 has 647/648 shards, and one retained non-special teacher-forced score is non-finite; no formal metric |
| Whisper numeric diagnostic | Same BF16-generated four sequences scored as BF16 batched, BF16 per-hypothesis and FP32-upcast batched; per-token finite classification; no cache writes | `scripts/diagnose_asrur_whisper_numeric.py`, `scripts/run_asrur_whisper_numeric_diagnostic.sh`; success audit `results/audits/asrur_whisper_numeric_diagnostic_success_20260728.json` | COMPLETE at execution `c294730`; wrapper=`0`, elapsed=`33s`, all three paths and beam sequence scores finite, peak allocated=`6,448,808,960` B |
| Whisper numeric diagnostic serialization failure | Strict JSON rejected a raw non-finite beam sequence score after all three scoring calls returned | `results/audits/asrur_whisper_numeric_diagnostic_serialization_failure_20260728.json` | FAILED at execution `a7735e4`; diagnostic output was not atomically published and no cache was changed; repaired representation is `null` plus an explicit numeric class |
| Phase-2 attempt-4 recovery entrypoint | Reuse 14 complete immutable caches, strictly validate/copy 647 partial snr_0 shards with source-commit provenance, and permit at most three same-protocol attempts only for classified non-finite Whisper failures | `scripts/run_asrur_phase2_attempt4.sh`, `scripts/generate_asrur_frozen_caches.py`, `tests/test_run_asrur_phase2_attempt4.py`, `tests/test_generate_asrur_frozen_caches.py` | COMPLETE at execution `d9baf22`; missing record succeeded on its first new attempt, no filtering, clamping, score replacement, mixed protocol, training, or download |
| Phase-2 attempt-4 formal outcome | Four-condition B1/B2/B3/U1 rankings, OEA fixed-candidate oracle, strict completion and Go/No-Go evidence | compact audit `results/audits/asrur_phase2_attempt4_no_go_20260728.json`; remote run `asrur_phase2_frozen_retrieval_execute_20260728_093950` | COMPLETE execution at `d9baf22`, but scientific decision=`NO_GO_REQUIRES_USER_DECISION`; 44/44 steps and wrapper exited 0, while OEA and Whisper routes are failure evidence rather than usable baselines |
| Phase-2 NO-GO diagnostic | CPU-only Whisper transcript/WER, N-best diversity, cross-condition agreement, OEA/vanilla embedding geometry, hubness, and positive-score characterization | `scripts/diagnose_asrur_phase2_no_go.py`, `tests/test_diagnose_asrur_phase2_no_go.py` | COMPLETE on 648 queries; current Whisper output is a universal `Thank you.` hallucination (Clean/20/10 dB unique Top-1=`1`, WER=`1.0`), while OEA projected embeddings are substantially more anisotropic than vanilla and cover only `3–6/648` positives at Top-100 |
| Whisper content differential diagnostic | Same deterministic SQuTR-FiQA records under current generic BF16 four-best, official Whisper BF16 one-best, and official Whisper FP32 one-best; audio signal statistics; no formal-cache writes | `scripts/diagnose_asrur_whisper_content.py`, `scripts/run_asrur_whisper_content_diagnostic.sh`, `tests/test_diagnose_asrur_whisper_content.py`, audit `results/audits/asrur_whisper_content_diagnostic_success_20260728.json` | COMPLETE at `576c4f6`; 8 records, wrapper=`0`, elapsed=`166s`; generic WER=`0.141304`, official BF16/FP32 WER=`0.130435`, proving the old universal-hallucination cache must be rebuilt rather than reused |
| Phase-2 attempt-5 clean Whisper recovery | Selective immutable reuse of 11 OEA/vanilla/BGE cache groups, zero Whisper reuse/import, fresh four-condition generation, and per-condition catastrophic-content integrity gate | `scripts/run_asrur_phase2_attempt5.sh`, `scripts/audit_asrur_whisper_cache.py`, `scripts/run_asrur_phase2_frozen_retrieval.sh` | IN_PROGRESS at execution `dc995b5`; Clean 648/648 first attempts passed the content gate with 648 unique Top-1 and WER `0.127663`; snr_20 followed |
| OEA collapse attribution diagnostic | Same deterministic FiQA audio/documents in base-hidden, LoRA-hidden, base-plus-OEA-heads, and full LoRA-plus-OEA-heads spaces; fresh/cache agreement | `scripts/diagnose_asrur_oea_collapse.py`, `scripts/run_asrur_oea_collapse_diagnostic.sh`, `tests/test_diagnose_asrur_oea_collapse.py`; compact audit `results/audits/asrur_oea_collapse_attribution_20260728.json` | COMPLETE at execution `dc995b5`; base hidden R@10=`1.0`, full LoRA+heads R@10=`0.25`; fresh full audio/cache cosine=`1.0`, so stale audio cache is not the global cause |
| OEA-Nemo3B-Cl FiQA independent full rerun | Same fixed checkpoint/protocol; fresh 57,638-document and 2,592-audio encoding; no old OEA embedding reuse; exact Top-100, metrics/oracle, row-level cache/metric comparison | `scripts/run_asrur_oea_cl_fiqa_rerun.sh`, `scripts/audit_asrur_oea_cl_fiqa_rerun.py`, `tests/test_audit_asrur_oea_cl_fiqa_rerun.py` | IMPLEMENTED_AND_TESTED; WAITING_USER for one RTX 4090 full-rerun approval; resumable inside its own new cache root, offline, no training/checkpoint selection |
| Phase-2 dry-run attempt 1 | CPU-only; OEA resolution passed, vanilla direct entrypoint import failed before model loading | `results/audits/asrur_phase2_dry_run_import_failure_20260727.json` | FAILED at execution `7a098dd`; original error preserved; minimal entrypoint fix passes 388 full tests |
| Phase-2 dry-run attempt 2 | CPU-only end-to-end command/config/input validation; 17 planned steps | `results/audits/asrur_phase2_dry_run_success_20260727.json` | COMPLETE at execution `1ff9fc0`; 17/17 steps, run, wrapper and caller all exited 0; stderr empty; no model/GPU/network use |
| Phase-2 normalization smoke attempt 1 | Direct real-model smoke gate after the float32 cache-boundary normalization repair | `results/audits/asrur_phase2_norm_smoke_entrypoint_failure_20260727.json` | FAILED at execution `92d705f` before model loading; 16 CPU tests and RTX 4090 preflight passed, but the direct generator entrypoint could not import `scripts`; Phase-2 was not launched |
| Phase-2 normalization smoke attempt 2 | Real-model validation of the direct-entrypoint and float32 cache-boundary normalization repairs | `results/audits/asrur_phase2_norm_smoke_success_20260727.json` | COMPLETE at execution `ee63772`; 5 audio + 25 text rows, 17/30 outside the former 1e-3 tolerance before repair, post-repair maximum norm deviation `1.19e-7`; Phase-2 attempt 2 launched afterward |
| Phase-2 live monitor | Read-only process/GPU/step/progress/metrics/completion view with follow and one-shot modes | `scripts/monitor_asrur_phase2.sh`, `tests/test_monitor_asrur_phase2.py` | IMPLEMENTED; reports run/step elapsed, local progress, recent throughput and approximate step ETA; future detached launches are tmux-only by user policy |
| Phase-2 result auditor | Strict four-condition method/query/path validation plus locked per-condition and conservative aggregate Go/No-Go | `scripts/audit_asrur_phase2_results.py`, `tests/test_audit_asrur_phase2_results.py` | IMPLEMENTED_AND_TESTED; does not change candidate generation and never converts a failed condition into automatic GO |
| Phase-3 unselected CE runner | One frozen four-hypothesis CE matrix plus a separate single-gold-text CE upper bound per acoustic condition; reports 1-best CE, 4-best equal, 4-best max and U2 Gold CE without FiQA-dev hyperparameter selection | `scripts/run_asrur_phase3_unselected_ce.sh`, `scripts/evaluate_asrur_unselected_ce_baselines.py`, `scripts/build_asrur_gold_nbest.py` | IMPLEMENTED_AND_TESTED; 28 related tests, Python compile and Bash syntax pass; gold has no fabricated ASR posterior; execution is gated on Phase-2 aggregate GO and a separate GPU approval |

最后更新：2026-07-28（Whisper attempt 5 运行中；OEA-Cl 全量独立重跑已实现并等待 GPU）

大型数据、权重、embedding、索引和日志不进入普通 Git。本表登记身份、路径、证据
和使用限制。`historical remote evidence` 表示已有完成证据，但正式运行前仍要做
实时只读复核。

## 1. 已有论文、代码和小型证据

| ID | 资产 | 身份/哈希 | 路径 | 状态 |
|---|---|---|---|---|
| P-OEA | OEA PDF | SHA256 `e76bbd96c82ac78568ed75eaedcbc62e389358c4bb226457ac45b55ab63cfe38` | `C:\论文\Omni-Embed-Audio.pdf` | LOCAL_VERIFIED |
| C-OEA | 当前代码 | Phase 0 前 HEAD `8d98c256...`，branch `repro/oea-full` | 当前仓库 | LOCAL_VERIFIED |
| C-SQUTR | SQuTR official code reference | commit `cc3fb31fc0dc44fef3a44c569b344516bbaee79c` | GitHub；不 vendored | METADATA_VERIFIED |
| A-SQUTR-13A | SQuTR archive audit | audit JSON committed | `results/audits/squtr_data13a_archive_audit_20260726.json` | COMPLETE |
| A-SQUTR-13B | SQuTR content audit | attempt 1 `0/1/1`；attempt 2 `0/1/1`；attempt 3 wrapper `20` | attempt 1/2 failure JSONs committed；attempt 3 lock failure: `results/audits/squtr_data13c_attempt3_lock_failure_20260726.json` | ATTEMPTS_1_2_FAILED; ATTEMPT_3_LOCK_DENIED |
| R-OEA4 | Qwen3B-Cl Clotho A2T | R@1/5/10 `27.2727/52.8230/66.6986` | `results/audits/qwen3b_cl_clotho_a2t_eval_20260721.json` | COMPLETE, not paper-reported |
| R-OEA6-A100 | Qwen3B-Cl efficiency | controlled A100 result | `results/audits/qwen3b_clotho_a100_efficiency_20260721.json` | COMPLETE |
| R-OEA6-4090 | Qwen3B-Cl efficiency | hardware-mismatched extension | `results/audits/qwen3b_clotho_rtx4090_efficiency_20260721.json` | COMPLETE |
| C-OEA-B6-NEMO-4090 | Selected Nemo3B-Cl efficiency protocol | immutable config | `configs/eval/nemo3b_cl_clotho_efficiency_rtx4090.json` | READY; execution requires separate GPU approval |
| S-OEA-B6-NEMO-4090 | Selected Nemo3B-Cl efficiency wrapper | lock-bound, offline, non-overwriting runner | `scripts/run_nemo3b_cl_clotho_efficiency_rtx4090.sh` | READY; use an isolated checkout while Phase 2 is active |
| C-ASRUR-RES | D1–D4 pinned manifests + resumable CPU wrapper | commit `6279b8265a3c90be92536eb96bcd98408bebfffa` | `configs/asr_uncertainty_reranking/resources/`、`scripts/run_asrur_resource_download.sh` | COMPLETE implementation; D1 later completed, old sequential D2–D4 run intentionally stopped |
| C-ASRUR-CORE | normalization、proxy posterior、4-best aggregation、metrics、cache manifest | commit `8cc5984dc35a272934d434993006814855f39624` | `AudioRetrieval/asr_uncertainty_reranking/` | COMPLETE; 38 related CPU tests passed |
| C-ASRUR-PIPE | B1–B7/QG/Ours/U1–U4/A1–A9 CPU 实验框架 | commit `8f28c5590810f3e430d3ce7591266ad7f8e38f21` | `AudioRetrieval/asr_uncertainty_reranking/`、`scripts/*asrur*`、`configs/asr_uncertainty_reranking/main_experiment.json` | COMPLETE; 58 ASRUR tests and 345 full-repository tests passed; no research result generated |
| A-MODEL-MIGRATION | OEA 模型迁移与旧缓存清理 | run `model_cache_migration_v3_20260726_214834`；exit `0`；254 files 前后相同 | `results/audits/model_cache_migration_20260726.json` | COMPLETE; `/home/jg525/model_cache` removed |
| C-ASRUR-MODEL-AUDIT | D2–D4 离线固定 revision/LFS/size/SHA256 验收器 | commit `aac088d68f9bc60871c48db90a2603144ca1b178` | `scripts/run_asrur_model_audit.sh` | COMPLETE；已执行首次远程验收 |
| F-ASRUR-D2-D4-AUDIT1 | D2–D4 首次严格离线验收 | execution `92746e5`；run `asrur_d2_d4_model_audit_20260727_132612`；audit/wrapper=`1/1` | `results/audits/asrur_d2_d4_model_audit_failure_20260727.json` | FAILED；D3/D4 complete，D2 仅缺 3,087,130,976-byte `model.safetensors`；未修改/删除模型 |
| A-ASRUR-D2-D4-AUDIT2 | D2 修复与 D2–D4 严格离线验收 | run `asrur_d2_whisper_repair_20260727_134626`；strict/wrapper=`0/0`；selected bytes=`5,823,662,271/5,823,662,271` | `results/audits/asrur_d2_d4_model_audit_success_20260727.json` | COMPLETE；Whisper 权重 3,087,130,976 B、SHA256 `a8e94b85...fd95`；D3/D4 不重复下载 |
| C-ASRUR-NEMO-AUDIT | Nemo Phase 1 CPU 三阶段审计与 portable lock runner | initial `953cb34`；metadata fixes `86107ca`/`6760f7f`；safe-global fix `7bd4765` | `scripts/run_asrur_nemo_phase1_audit.sh`、`scripts/run_official_oea_model_pipeline.py` | COMPLETE implementation；CPU-only；固定 revision metadata API only、无模型下载；attempt 4 待用户操作 |
| F-ASRUR-NEMO-METADATA | Nemo Phase 1 attempt 1 metadata-policy failure | execution `d3ba9f1`；run `asrur_nemo_phase1_audit_20260726_232452`；exit `1/1` | `results/audits/asrur_nemo_phase1_metadata_policy_failure_20260726.json` | FAILED；实现协议冲突，不是模型损坏；由 `86107ca` 修复 |
| F-ASRUR-NEMO-TRANSITIVE-OFFLINE | Nemo Phase 1 attempt 2 legacy offline-alias failure | execution `5bba16c`；run `asrur_nemo_phase1_audit_20260726_235146`；exit `1/1` | `results/audits/asrur_nemo_phase1_transitive_offline_failure_20260727.json` | FAILED；`TRANSFORMERS_OFFLINE` 被 Hub 0.36.0 视为离线别名；由 `6760f7f` 修复 |
| F-ASRUR-NEMO-POSIXPATH | Nemo Phase 1 attempt 3 safe-global compatibility failure | execution `cd34cbc`；run `asrur_nemo_phase1_audit_20260727_000937`；resource/preparation=`0/1` | `results/audits/asrur_nemo_phase1_posixpath_alias_failure_20260727.json` | FAILED；资源全量通过，`pathlib._local.PosixPath` 精确安全别名缺失；由 `7bd4765` 修复 |
| A-ASRUR-NEMO-PHASE1 | Nemo Phase 1 attempt 4 CPU gate success | execution `58f80bb`；run `asrur_nemo_phase1_audit_20260727_004107`；pipeline/wrapper=`0/0` | `results/audits/asrur_nemo_phase1_success_20260727.json` | COMPLETE；resource/preparation/lock 均 complete |
| L-OEA-NEMO-CL | OEA-Nemo3B (+Cl) canonical model lock | 5,932 B；SHA256 `fd09e4d2...c8c2` | `results/model_locks/oea_nemo3b_cl.json` | LOCKED；与远程 portable lock 逐字一致 |
| L-VANILLA-NEMO | Original Omni-Embed-Nemotron canonical model lock | 6,416 B；SHA256 `eb62da75...cc1c` | `results/model_locks/vanilla_nemotron_3b.json` | LOCKED；与远程 portable lock 逐字一致；仅声明 base snapshot 与 `[CODE]` pooling/prompt 协议 |
| C-ASRUR-NEMO-G1 | Nemo3B-Cl Clotho 锁绑定 5/25 smoke、同锁全量生成与 CPU T2A 套件 | generator commit `ff73287`；suite commit `64647a0`；T2A 套件绑定同一 model lock 与生成协议 SHA256 | `configs/eval/nemo3b_cl_clotho_*`、`scripts/run_nemo3b_cl_clotho_*` | COMPLETE implementation；Nemo 本地 custom model code 由完整模型锁逐文件约束 |
| R-ASRUR-NEMO-G1-SMOKE | Nemo3B-Cl Clotho 锁绑定 5/25 GPU smoke | execution `1b23c50`；run `..._20260727_092654`；RTX 4090；exit `0/0` | `results/audits/asrur_nemo_g1_smoke_20260727.json`；大型原始工件保留远程 run 目录 | COMPLETE；5×512/25×512、544 LoRA、严格离线、峰值 9.14/9.49 GiB；不是论文 Recall 结果 |
| R-ASRUR-NEMO-G1-FULL | Nemo3B-Cl Clotho 全量 embedding | execution `4d2341d`；run `..._20260727_094554`；RTX 4090；exit `0/0` | `results/audits/asrur_nemo_g1_full_embeddings_20260727.json`；大型 embedding/chunks 保留远程 run 目录 | COMPLETE；1,045×512 audio、5,225×512 text、严格离线、峰值 9.39/10.48 GiB；Recall 尚待 CPU 套件 |
| R-ASRUR-NEMO-G1-T2A | Nemo3B-Cl Clotho Table 2 T2A | execution `20533b1`；suite `..._20260727_125803`；CPU；suite/attempt/run=`0/0/0` | `results/audits/asrur_nemo_clotho_t2a_20260727.json`；suite metrics SHA256 `c6121e7c...bde0`；CSV SHA256 `6a192065...ed8` | COMPLETE；all-caption 最大绝对差 0.152488 pp，seed0 最大差 0.551388 pp；两组均为 `[CODE] close`，严格论文 caption 口径仍 blocked |
| R-ASRUR-NEMO-G1-A2T | Nemo3B-Cl Clotho A2T direction gate | execution `7cfbfab`；suite `..._20260727_161133`；CPU；attempt=`0`、stderr 为空 | `results/audits/asrur_nemo_clotho_a2t_20260727.json`；suite metrics SHA256 `c54994a9...cc60`；CSV SHA256 `37a73e11...8e5` | COMPLETE；1,045 audio→5,225 captions；R@1/5/10=`26.8900/51.3876/65.2632`；`[CODE]` extension, not paper-reported |

## 2. 已有远程数据

| ID | 数据 | 固定身份 | 远程路径 | 状态/限制 |
|---|---|---|---|---|
| D-SQUTR-ZIP | SQuTR full archive | revision `2f1b041...`; license `CC BY-SA 4.0`; 21,069,841,248 B; SHA256 `8956bf...c6997c` | `/home/jg525/datasets/oea/squtr/source/source_data.zip` | COMPLETE; contains `en/fiqa` and `en/nq` |
| D-SQUTR-EXTRACT | SQuTR extracted tree | 149,310 files；28,422,366,590 member bytes；full CRC | `/home/jg525/datasets/oea/squtr/extracted` | EXTRACTION_COMPLETE; do not consume before content gate |
| D-SQUTR-MANIFEST | 149,268-query audio manifest | generated by DATA-13B | `/home/jg525/datasets/oea/squtr/manifests/squtr_audio_query_manifest.jsonl` | PENDING |
| D-SQUTR-FIQA-NQ-MANIFEST | 16,400 audio instances：FiQA 648 + NQ 3,452，四条件 | 15,693,052 B；SHA256 `e5053d30...15e93`；`record_id`/`sample_id` compatibility checked | `/home/jg525/datasets/oea/squtr/manifests/squtr_en_fiqa_nq_audio_query_manifest.jsonl` | COMPLETE；DATA-13D exit `0/0/0/0` |
| D-SQUTR-FIQA-NQ-PROBE-CACHE | 16,400 resumable first/last-frame target-subset audio probes | 16,401 lines including identity header；SHA256 `8683dc72...169c` | `/home/jg525/datasets/oea/squtr/manifests/.squtr_audio_probe_cache_en_fiqa_nq_v1.jsonl` | COMPLETE |
| A-SQUTR-13D | FiQA/NQ content and FiQA identity audit | commit `730e0fc`；run `data13d_squtr_fiqa_nq_validation_20260726_224653` | `results/audits/squtr_data13d_fiqa_nq_validation_20260726.json` | COMPLETE；no violations or split leakage |
| D-SQUTR-PROBE-CACHE-V1 | attempt-2 identity-only audio probe cache | commit `68aa615` bound；475 bytes；1 line；SHA256 `29b63e98...e6446` | `/home/jg525/datasets/oea/squtr/manifests/.squtr_audio_probe_cache_v1.jsonl` | PRESERVED_ATTEMPT_EVIDENCE |
| D-SQUTR-PROBE-CACHE-V2 | resumable first/last-frame audio probe cache | marker/structure/protocol/root/Git-bound | `/home/jg525/datasets/oea/squtr/manifests/.squtr_audio_probe_cache_v2.jsonl` | NOT_CREATED; attempt 3 exited before validation |
| D-CLOTHO-EVAL | Clotho v2.1 evaluation | prior DATA-02 validated | `/home/jg525/datasets/oea/clotho_v2.1` | COMPLETE evidence; live recheck before Phase 1 |
| D-FIQA-MTEB | FiQA train/dev/test metadata | revision `5e59eeb...`; selected 48,616,245 B | `/home/jg525/datasets/oea/fiqa_mteb` | COMPLETE; run `asrur_d1_fiqa_download_20260726_200414`, exit `0/0` |
| D-NQ-MTEB | NQ public snapshot | revision `b84726e...` | `/home/jg525/datasets/oea/nq_mteb` | DEFERRED; SQuTR copy may suffice |

## 3. 已有与计划模型

| ID | 模型 | revision / primary file | 远程路径 | 状态 |
|---|---|---|---|---|
| M-NEMO-BASE | `nvidia/omni-embed-nemotron-3b` | `865db1bb...` | `/home/jg525/models/oea/omni-embed-nemotron-3b` | LIVE_CONTENT_VERIFIED；22 files / 9,423,120,401 B，attempt 3 |
| M-OEA-NEMO-AC | `JudeJiwoo/OEA-Nemo3B-AC` | `8ed66aa...`; `step_400_best.pt` | `/home/jg525/models/oea/OEA-Nemo3B-AC` | migrated intact by file-count/byte invariants |
| M-OEA-NEMO-CL | `JudeJiwoo/OEA-Nemo3B-Cl` | `9588912...`; `step_450_best.pt` | `/home/jg525/models/oea/OEA-Nemo3B-Cl` | LIVE_CONTENT_VERIFIED；3 files / 9,466,834,755 B；source SHA256 `c9013285...a96d`；59,072,047-B derived SHA256 `2a5bee90...80c4` |
| M-WHISPER | `openai/whisper-large-v3` | `06f233fe...`; weight 3,087,130,976 B；SHA256 `a8e94b85...fd95` | `/home/jg525/models/whisper-large-v3` | COMPLETE；strict offline audit passed |
| M-BGE-DENSE | `BAAI/bge-base-en-v1.5` | `a5beb1e3...`; selected snapshot bound by the D2–D4 manifest | `/home/jg525/models/bge-base-en-v1.5` | COMPLETE；strict offline audit passed |
| M-BGE-RERANK | `BAAI/bge-reranker-v2-m3` | `953dc6f6...`; selected snapshot bound by the D2–D4 manifest | `/home/jg525/models/bge-reranker-v2-m3` | COMPLETE；strict offline audit passed |
| M-TTS | FiQA train/dev TTS model | not selected | not assigned | BLOCKED pending protocol/license approval |

## 4. 将生成的缓存

以下路径是规划，不代表文件已经存在：

| 缓存 | 计划根目录 | Manifest 必须包含 |
|---|---|---|
| OEA/Omni document embedding | `/home/jg525/experiment_cache/asr_uncertainty/document_embeddings` | corpus checksum、text construction、model lock、dim、dtype、normalization |
| Audio query embedding | `/home/jg525/experiment_cache/asr_uncertainty/audio_embeddings` | SQuTR manifest checksum、condition、audio hash、model lock |
| BGE embedding/index | `/home/jg525/experiment_cache/asr_uncertainty/bge` | query instruction、max length、pooling、revision、index params |
| Whisper N-best | `/home/jg525/experiment_cache/asr_uncertainty/whisper_nbest` | decoding config、raw scores、proxy posterior definition、failed IDs |
| OEA Top-100 | `/home/jg525/experiment_cache/asr_uncertainty/oea_top100` | exact corpus/query/qrels hash、candidate ordering、tie policy |
| CE scores | `/home/jg525/experiment_cache/asr_uncertainty/cross_encoder` | Top-100 hash、N-best hash、reranker revision、max length |
| Gate features | `/home/jg525/experiment_cache/asr_uncertainty/gate_features` | all upstream manifest hashes、feature schema/version |
| Runs | `/home/jg525/Omni-Embed-Audio/results/raw/<run_id>` | config、command、stdout/stderr、metrics、environment、commit、GPU |

缓存 manifest 不匹配时直接拒绝复用，不做隐式升级或覆盖。

## 5. 已锁定的 CPU 定义

- z-score 使用每条 query 候选集合内的 population standard deviation；退化路由输出全零。
- rank normalization 对同分候选使用平均秩；需要完整排序时以候选 ID 做确定性 tie-break。
- Whisper N-best 仅生成 `proxy posterior`，不宣称严格声学后验。
- 4-best 聚合固定支持 one-best、等权均值、最大值和
  `logsumexp(log p_m + r_mi / T_ce)`。
- nDCG 使用全局 qrels 的 graded ideal ranking；Oracle 只重排固定候选集合，不补入漏召回文档。
- 指标内部值统一为 `[0,1]` fraction；最终制表时才显式转为百分比。
- 缓存身份默认比较数据/输入哈希、模型 revision/checkpoint、tokenizer、pooling、维度、长度、dtype、归一化、seed 和 producer Git commit。跨 commit 复用必须显式允许并记录，不能静默发生。
- B5/B6 主融合路由已实现为可锁配置；`[INFERRED][USER-CONFIRMED
  2026-07-26]` 正式主路线固定为 4-best `proxy_posterior`，1-best 只保留为
  辅助诊断。不得在看到 FiQA test 或 NQ 结果后切换该路线。
- Whisper proxy 的实现口径已进一步锁定：4-best 候选仍来自冻结 beam search，
  beam `sequence_score` 作为排序证据单独保存；用于 proxy posterior 的
  `average_token_logprob` 来自对 exact generated sequence 的冻结
  teacher-forced float32 cross-entropy，不使用 generation transition scores，
  不宣称是校准的 ASR posterior。失败 attempt 2 证据：
  `results/audits/asrur_whisper_transition_score_smoke_failure_20260727.json`。
