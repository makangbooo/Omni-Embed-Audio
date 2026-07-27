# ICASSP experiment checklist

Last updated: 2026-07-27

This is the scope of the proposed paper:

> OEA is the frozen first-stage spoken-query retriever. Whisper N-best
> uncertainty and a frozen BGE cross-encoder provide lexical evidence. A
> small candidate-level gate learns how much to trust each source.

It is **not** a full reproduction of every experiment in the OEA paper. An
item is complete only when a real-data result and its auditable artifacts
exist; implemented code or a synthetic test is not counted as a completed
experiment.

Allowed statuses are `TODO`, `IN_PROGRESS`, `WAITING_USER`,
`RUNNING_REMOTE`, `COMPLETED`, `FAILED`, and `BLOCKED`.

## A. OEA experiments required for this paper

| ID | Required result | Why it is required | Status | Evidence / blocker |
|---|---|---|---|---|
| OEA-B0 | Lock OEA-Nemo3B(+Cl), its base model, prefix, pooling, projection, dimension, and normalization; run a real embedding smoke test | Prevents an invalid OEA baseline | COMPLETED | Canonical lock and RTX 4090 5-audio/25-text smoke; peak allocated 9.14 GiB |
| OEA-B1 | Reproduce OEA-Nemo3B(+Cl) Clotho T2A R@1/5/10 | Numerical connection to the OEA paper | COMPLETED | Paper 21.57/47.16/60.36; reproduced all-caption 21.7225/47.1196/60.4402 |
| OEA-B2 | Clotho audio-to-text direction sanity check using the selected Nemo checkpoint | Confirms the direction used by the new task before changing domains | COMPLETED | CPU suite `..._20260727_161133` completed at `7cfbfab`: 1,045 audio queries against 5,225 captions; R@1/5/10=`26.8900/51.3876/65.2632`; stderr empty; this is a `[CODE]` direction extension, not a paper-reported result |
| OEA-B3 | SQuTR-FiQA OEA direct audio-to-text retrieval for Clean/20/10/0 dB | Primary frozen first-stage baseline and candidate generator | WAITING_USER | OEA caches remain complete; the failed snr_0 Whisper record passed all three isolated numeric paths on retry. Attempt-4 recovery is tested and awaits one RTX 4090; no formal ranking metric exists yet |
| OEA-B4 | SQuTR-FiQA original `nvidia/omni-embed-nemotron-3b` direct retrieval for four conditions | Shows whether OEA adaptation is a meaningful baseline | WAITING_USER | All four vanilla caches are complete and checksummed for reuse; no formal ranking metric exists until repaired Phase 2 resumes |
| OEA-B5 | OEA Top-100 Recall@20/50/100 and Top-100 Oracle nDCG@10 | Establishes whether reranking can succeed without changing recall | WAITING_USER | Depends on repaired Phase-2 finalization; no metric is complete |
| OEA-B6 | Selected Nemo OEA audio/text latency, throughput, and peak memory on the study GPU | Measures the cost inherited by the proposed system | WAITING_USER | RTX 4090 lock-bound entrypoint is ready; requires a separate 1-GPU approval and an isolated checkout while Phase 2 is running |

The following OEA-paper experiments are outside the ICASSP claim and will not
consume compute unless the paper scope changes: all six OEA variants, all
AudioCaps/MECAT/UIQ/T2T grids, negative-query and hard-negative tables, CLAP
baselines, full OEA training, leakage reprocessing, and every backbone/efficiency
combination. Existing results for those tasks remain preserved but do not count
toward this checklist.

## B. Additional experiments required by the innovation

| ID | Required result | Status | Evidence / blocker |
|---|---|---|---|
| ASR-E1 | B1: Whisper 1-best + BGE dense retrieval, four FiQA conditions | WAITING_USER | clean/snr_20/snr_10 complete; snr_0 has 647/648 shards. Exact-record BF16/FP32 numerical diagnosis is required before a non-fabricated repair |
| ASR-E2 | B4: OEA Top-100 + 1-best cross-encoder | BLOCKED | Formal no-test-selection evaluator and resumable four-condition runner are tested; execution waits for Phase-2 aggregate GO and separate GPU approval |
| ASR-E3 | B5: fixed fusion using preregistered 4-best proxy-posterior ASR evidence | BLOCKED | Dev selection only; no test tuning |
| ASR-E4 | B6: RRF using the same 4-best proxy-posterior ASR route | BLOCKED | Dev selection only; no test tuning |
| ASR-E5 | B7a: 4-best equal aggregation | BLOCKED | Shares the tested four-hypothesis CE cache with E2; execution waits for Phase-2 aggregate GO and separate GPU approval |
| ASR-E6 | B7b: 4-best maximum aggregation | BLOCKED | Shares the tested four-hypothesis CE cache with E2; execution waits for Phase-2 aggregate GO and separate GPU approval |
| ASR-E7 | B7c: 4-best proxy-posterior aggregation | BLOCKED | Needs Whisper N-best and CE cache |
| ASR-E8 | Query-level uncertainty gate baseline | BLOCKED | Needs FiQA train/dev generated speech and frozen upstream caches |
| ASR-E9 | Ours: candidate-level dynamic gate, three seeds | BLOCKED | Needs approved TTS/noise protocol and gate training |
| ASR-E10 | U1: gold text query + BGE dense upper bound | WAITING_USER | BGE corpus/dev caches are complete and reusable; gold-test encoding and final metric resume after the Whisper repair smoke |
| ASR-E11 | U2: gold transcript + frozen cross-encoder upper bound | BLOCKED | Strict one-hypothesis Gold artifact and same-candidate CE/evaluator are tested without fabricating an ASR posterior; execution waits for Phase-2 aggregate GO and separate GPU approval |
| ASR-E12 | U3/U4: candidate oracle and candidate Recall@20/50/100 | WAITING_USER | Requires repaired Phase-2 ranking finalization; no metric is complete |
| ASR-E13 | Main FiQA table: all required methods × four acoustic conditions | BLOCKED | Depends on ASR-E1 through ASR-E12 |
| ASR-E14 | Three-seed gate mean, standard deviation, and 95% confidence interval | BLOCKED | Depends on ASR-E9 |
| ASR-E15 | A1–A9 ablations | BLOCKED | Evaluation code is tested; real caches/results absent |
| ASR-E16 | Paired bootstrap against the strongest baseline | BLOCKED | Tested implementation; real per-query metrics absent |
| ASR-E17 | WER and performance stratified by WER | BLOCKED | Needs Whisper outputs and formal rankings |
| ASR-E18 | Gate weight versus SNR/ASR uncertainty; verify lower trust under unreliable ASR | BLOCKED | Needs trained gates and formal features |
| ASR-E19 | Complementarity and failure-case analysis | BLOCKED | Needs final per-query rankings |
| ASR-E20 | End-to-end and component latency, throughput, peak memory | BLOCKED | Needs correctness-complete models and separate GPU approval |
| ASR-E21 | NQ cross-domain zero-shot main methods and upper bounds | BLOCKED | Conditional on all six preregistered FiQA Go/No-Go requirements |

## C. Execution prerequisites

| ID | Prerequisite | Status | Evidence / next action |
|---|---|---|---|
| P0 | Protocol, paths, splits, qrels, no-test-tuning rules | COMPLETED | Main config and Phase-0 audit locked |
| P1 | SQuTR `en/fiqa` and `en/nq` extraction/validation | COMPLETED | 648×4 FiQA and 3,452×4 NQ audio rows validated |
| P2 | FiQA corpus/train/dev/test/qrels | COMPLETED | 57,638/5,500/500/648 query protocol validated |
| P3 | OEA checkpoint/base resources | COMPLETED | Fixed revisions and checksums locked |
| P4 | D2 Whisper-Large-v3 | COMPLETED | Repair run `..._20260727_134626`; strict/wrapper=`0/0`; 3,087,130,976-byte weight SHA256 `a8e94b85...fd95`; full D2–D4 selected bytes `5,823,662,271/5,823,662,271` |
| P5 | D3 BGE-base-en-v1.5 | COMPLETED | Fixed revision passed strict local audit |
| P6 | D4 BGE-reranker-v2-m3 | COMPLETED | Fixed revision passed strict local audit |
| P7 | Dependency-light ranking/gating/statistics framework | COMPLETED | Synthetic tests only; no synthetic number is a research result |
| P8 | Real BGE/Whisper/CE/OEA/Omni cache producers | COMPLETED | Resumable immutable caches; one shared document cache/model; strict FiQA qrels-ID and dev/test isolation; dev-only BGE template selection; both OEA and vanilla canonical locks; exact one-RTX-4090 end-to-end wrapper; exact rankings and candidate oracle; 17/17-step CPU dry-run passed at `1ff9fc0`; GPU execution is not yet authorized |
| P9 | FiQA train/dev TTS, speaker split, and noise construction | WAITING_USER | Deliberately postponed until Phase-2 Go/No-Go; requires a separate concrete proposal |

## D. Progress accounting

- Required experiment packages: **28** (`OEA-B0`–`OEA-B6` plus
  `ASR-E1`–`ASR-E21`).
- Completed experiment packages: **3/28 = 10.7%**.
- Completed main FiQA result cells: **0**. Code readiness must not be reported
  as an experimental result.
- Prerequisites completed: **9/10**; the remaining TTS/noise prerequisite
  deliberately awaits the later Phase-2 Go/No-Go decision.
- A transparent project-level estimate is **30%**:
  preparation is weighted 30% and approximately 90% complete; real experiments
  are weighted 60% and approximately 5% complete; final analysis/reporting is
  weighted 10% and 0% complete. This estimate is planning metadata, not a
  scientific result.

The next critical path is:

1. diagnose and repair the single `en/fiqa:snr_0:10639` numerical failure without filtering or replacing its score, then resume Phase 2 from preserved caches;
2. complete OEA-B3/B4/B5 and ASR-E1/E10/E12;
3. apply the preregistered Go/No-Go without changing candidate generation;
4. if Go, create the frozen CE/N-best caches and run Phase 3;
5. separately approve the FiQA train/dev TTS/noise protocol before gate
   training.
