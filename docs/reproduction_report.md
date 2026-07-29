# OEA official-checkpoint reproduction report

This report is updated only from committed code and auditable run artifacts.
Environment setup, downloads, and unexecuted implementations are not counted
as reproduced experiments.

## OEA paper-only takeover snapshot

This section is the completion authority for the current task. SpeechXBT-OEA,
FiQA, NQ, SQuTR, ASR reranking, A2T, and other extensions later in this file
are retained only as historical evidence and do not contribute to OEA paper
completion.

At the current committed-evidence snapshot, the 31-row inventory remains
`4 COMPLETED / 10 IN_PROGRESS / 5 TODO / 12 BLOCKED`, or `4/31 = 12.9%`
complete under the whole-row inventory definition.
FIG-02 is now complete as a method and tensor-contract verification: independent
Qwen3B and Nemo3B lock-bound runs loaded the LoRA and dual-head contract and
produced 512-dimensional audio/text embeddings. This does not claim Qwen7B
runtime completion. The exact boundary and file hashes are recorded in
`results/audits/fig02_architecture_verification_20260728.json`.

The 910 visually audited paper metrics are aggregated into 374
experiment-model-dataset-task cells in
`results/tables/paper_experiment_matrix.csv`. Its `paper_status=PAPER` column is
separate from `REPRODUCED_CLOSE`, `CONTROLLED_ONLY`, `PARTIAL`, `TODO`, and
`BLOCKED`; the generation audit records all input/output hashes. The compact
paper-only partial Tables 2/3/12-15 are in
`results/tables/oea_partial_tables.md`. They contain 32 predeclared protocol
groups and do not select a protocol after comparing it with the paper value.

Committed value evidence currently covers four Clotho T2A cells, four Clotho
T2T cells, and sixteen Clotho positive-UIQ cells. OEA-Nemo3B-AC is the newest
complete paper cell: its official-source GPU precomputation and CPU finalizer
completed at run commits `effe67d` and `1f991ab`. The finalizer returned
`FINAL_RUN_RC=0` in 7 seconds and fixed eight protocols in
`results/audits/oea_nemo3b_ac_clotho_official_source_eval_20260729.json`.
The `[CODE]` all-caption/seed-0 T2A results are
`18.2775/41.3206/54.2010` and `17.9904/41.6268/55.7895`; the `[CODE]` seed-0
and `[INFERRED]` all-caption T2T results are `63.3493/74.8325/80.0957` and
`64.6316/74.9856/79.2536`. Released positive UIQ gives
Question=`18.9474/43.6364/56.3636`, Imperative=`19.0431/43.3493/56.8421`,
Paraphrase=`20.7656/44.3062/56.6507`, and
Keyphrase/tagging=`21.5311/45.8373/60.1914`. The failed non-empty-output UIQ
attempt and failed import-path metric attempt remain recorded; neither was
deleted or rewritten. Baseline NPZ SHA256 identities were not present in the
returned terminal excerpt and remain explicitly missing, while the successful
suite JSON SHA256 is fixed.

The earlier Nemo3B(+Cl) Clotho T2T cell completed at run commit `23bf6ee`:
the `[CODE]` seed-0 one-caption
protocol gives R@1/5/10=`62.9665/75.9809/80.8612`, while the predeclared
`[INFERRED]` all-caption sensitivity gives `63.6938/75.2344/80.0191`, against
paper values `63.77/75.29/80.11`. Both remain separate; the sensitivity result
is not promoted post hoc because it is closer. The suite finished with
`FINAL_RUN_RC=0` in 47 seconds, and its compact evidence is fixed in
`results/audits/oea_nemo3b_clotho_t2t_20260729.json`. Strict Table 3 protocol
status remains blocked because caption selection, self-exclusion, and tie
handling are not disclosed. Nemo3B(+Cl) positive UIQ subsequently completed on
an RTX 4090 at run commit `e3cce6f`: all 4,180 queries were embedded as 512-D
vectors, pending count is zero, the four CPU retrieval protocols are complete,
and all six discovered attempt/protocol exit-code files contain zero. The
released `[CODE]` protocols give Question=`23.6364/50.8134/63.9234`,
Imperative=`24.0191/51.1005/64.9761`, Paraphrase=`23.5407/49.9522/64.4019`,
and Keyphrase/tagging=`25.8373/52.6316/66.1244`; each category's maximum
absolute difference from PAPER is below 0.29 percentage points. They remain
`[CODE] close`, rather than strict paper-protocol results, because the public
audio candidate path omits PAPER's `passage:` prefix. Qwen3B(+Cl) efficiency and
14.71488M LoRA-plus-head parameter count are `CONTROLLED_ONLY`: strict Tables
5/16 remain blocked by the paper's missing timing, memory, count, and checkpoint
variant definitions.

The remote shared directories were inventoried read-only through the user's
authenticated Bitahub Web console. That check established that the current
instance `bitahub-a20633967503994880812290` uses ED25519 fingerprint
`SHA256:+lMykBVk8nCgA/Aav7/pG5AcS6UrJqy5OuKKLt+q8ZA`, while the anomalous
`etC2...` fingerprint does not match that instance. The new external endpoint
`xj-member.bitahub.com:42156` is not yet bound in the client `known_hosts`, and
no external connection was made by Codex. The newly completed Nemo T2T CPU run
is promoted only from its returned status, metrics, hashes, and remote paths.
During that read-only takeover audit, no download, GPU run, training, deletion,
or overwrite was started. The separately approved data runs are recorded below.

On 2026-07-29 the user approved DATA-04, DATA-06, DATA-08, and the
Nemo3B-Cl/Clotho positive-UIQ GPU task. DATA-04/05 and DATA-06/07 completed
through visible tmux wrappers at run commit `febfbe4`; DATA-08/09 then completed
at run commit `1046641`, followed by DATA-11 at `e415c1d`. The GPU task later
completed on a replacement RTX 4090 instance at run commit `e3cce6f`. The approvals are recorded in
`results/audits/takeover_approvals_20260729.json`. The user
requested an independent Bitahub-side host-key check and no client connection
in that turn; the read-only procedure is fixed in
`docs/ssh_host_key_verification.md`.

The MECAT run completed in 121 seconds with all four child/wrapper exit codes
equal to zero. It fixed the 173,168,424-byte archive at SHA256
`644cf75e...122c4`, decoded all 848 FLAC files, preserved all six caption
fields, and proved exact 848-ID equality for all four released positive UIQ
sets. The recursive count of 849 JSON files is 848 sample metadata files plus
the validator's hidden completion marker, not an extra sample. The canonical
manifest SHA256 is `b4c4d8c1...94a6`; full compact evidence is in
`results/audits/data04_data05_mecat_remote_20260729.json`. This does not unblock
strict MECAT Tables 2/3/12-15: PAPER's excluded 1 sample and the retrieval
caption construction remain `[MISSING]`.

The AudioCaps metadata wrapper completed in 40 seconds with
`FINAL_RUN_RC=0`. Subsequent read-only collection fixed all four child/wrapper
exit codes at zero, both evidence JSON hashes, and the line count, byte size,
and SHA256 of all four manifests. The public-loader and bare-CR-repaired train
manifests both contain 91,254 rows, so the paper's 91,256-row protocol remains
`[MISSING]`. Complete evidence is in
`results/audits/data06_data07_audiocaps_remote_20260729.json`; actual AudioCaps
evaluation audio is still absent.

The WavCaps metadata wrapper `oea_takeover_data08_20260729_124908` completed in
130 seconds with `FINAL_RUN_RC=0`. It verified exactly 8 pinned files totaling
176,863,095 bytes and explicitly excluded every audio archive. DATA-09 completed
the 403,050-row metadata/count/leakage pass and retained the protocol conflict:
PAPER writes `duration <= 31`, whereas the public metadata reaches 275,618 only
under `[INFERRED] 0 < duration < 31`. The wrapper evidence is fixed in
`results/audits/data08_data09_wavcaps_remote_20260729.json`. Subsequent read-only
collection fixed all four child/wrapper exit codes, both evidence JSON identities,
all eight source identities, and the line count, byte size, and SHA256 of the
403,050-row manifest plus four blocklist/candidate artifacts. These hashes match
the prior two-pass local audit. No exact post-blocklist paper training manifest
is claimed.

DATA-11 then completed the canonical MECAT/WavCaps source-video join with
call/audit/wrapper exit codes `0/0/0`. Among 848 public MECAT samples representing
807 unique source videos, it found four exact WavCaps AudioSet_SL source-video
candidates. The statistics SHA256 is `1e89bf60...630a` and the 4-row candidate
JSONL SHA256 is `d0f50a93...5b85`. This proves only common source-video IDs, not
temporal or audio-content overlap; EXP-09 remains `IN_PROGRESS` because the
paper's 847-row subset, embedding model, threshold, and reviewed candidates are
`[MISSING]`. Evidence is fixed in
`results/audits/data11_mecat_wavcaps_remote_20260729.json`.

The first Nemo3B-Cl/Clotho positive-UIQ preflight correctly stopped on a
CPU-only instance and is fixed in
`results/audits/nemo3b_cl_uiq_gpu_preflight_20260729.json`. The approved task was
then run on a replacement instance exposing one RTX 4090. Generation ran from
`2026-07-29T05:59:10Z` to `06:07:34Z` and produced a `[4180, 512]` embedding
array with zero pending chunks. The finalized CPU suite completed four 1,045-row
protocols. Compact evidence and all top-level artifact identities are fixed in
`results/audits/nemo3b_cl_clotho_positive_uiq_embeddings_20260729.json` and
`results/audits/nemo3b_cl_clotho_positive_uiq_eval_20260729.json`; large
embeddings and rankings remain at their recorded remote paths. The retained
tmux `combined.log` was classified as binary by `grep`, so its outer
`FINAL_RUN_RC` line was not returned and is not reconstructed. Completion is
instead established by the embedding attempt exit 0, suite/protocol exits 0,
finalized `complete` statuses, exact counts, and artifact hashes.

## Excluded SpeechXBT-OEA extension evidence

| ID | Evaluation | Source | Status | Evidence |
|---|---|---|---|---|
| OEA-4 | Clotho A2T: 1,045 audio queries against a frozen bank of 5,225 captions | `[CODE]` public A2T runner; not paper-reported | COMPLETED | CPU suite exit 0; R@1/5/10 = 27.2727/52.8230/66.6986; audit JSON and remote artifact hashes fixed |
| OEA-5 | SQuTR six-subset OEA zero-shot A2T over frozen text indexes | `[CODE]` official SQuTR dataset/retrieval schema plus explicit reproduction protocol | IN_PROGRESS | Immutable archive and DATA-13A structure audit complete; DATA-13B content audit and formal checkpoint run remain pending |
| OEA-6 | Audio-query encoding latency, peak memory, and throughput | `[PAPER]` A100 reference plus separately labelled hardware runs | COMPLETED | A100 and RTX 4090 controlled benchmarks complete; strict Table 5/16 protocol remains blocked by unpublished definitions |

## OEA-4 protocol

- Model: official `OEA-Qwen3B (+Cl)` checkpoint locked by
  `results/model_locks/oea_qwen3b_cl.json`.
- Dataset: Clotho v2.1 evaluation, exactly 1,045 audio queries and 5,225
  caption candidates.
- Candidate bank: the already generated caption embeddings are treated as a
  frozen text index; they are never regenerated by the CPU evaluator.
- Positives: all five captions whose exact `clip_id` equals the audio query's
  exact `candidate_id`.
- Similarity: L2-normalized dot product (cosine similarity).
- Rank: best-positive rank under the public code's optimistic
  strict-greater tie policy; full rankings use candidate index as the
  deterministic tie breaker.
- Metrics: R@1, R@5, R@10, MRR, and DCG.
- Claim boundary: OEA does not report A2T in a paper table. The result is an
  official-checkpoint `[CODE]` extension, not a reconstructed paper value.

The formal CPU run completed at Git commit
`b4986d3b129e97911fde93290f57dd55e4b21139` with an empty worktree status and
exit code 0. It evaluated all 1,045 audio queries against all 5,225 frozen
caption candidates and produced R@1 = 27.2727, R@5 = 52.8230, R@10 = 66.6986,
MRR = 0.395682, and DCG = 0.517412. The protocol `stderr.log` is empty. The
summary CSV SHA256 is `8c58983a...7e7fc`, and the finalized suite-metrics
SHA256 is `fd38d78c...8351`. Complete small evidence is retained in
`results/audits/qwen3b_cl_clotho_a2t_eval_20260721.json`; large similarities,
rankings, and embeddings remain at the recorded remote paths with fixed hashes.

An auxiliary precheck printed a `FileNotFoundError` because it expected an
embedding-root `metrics.json` that this generator does not create. This did
not affect the formal suite: the suite independently verified the actual
embedding-generation status and input hashes, finalized all required audit
artifacts, emitted empty stderr, and returned exit code 0.

## OEA-5 data and protocol audit

The target corpus is now fixed to the immutable SQuTR release at Hugging Face
revision `2f1b041e2e98e0d28ed68fbcf22126ef247eb719`. DATA-12 downloaded its
single 21,069,841,248-byte archive and verified SHA256
`8956bf938de3f9ce168a1e7daf2ff61b0b7fe603fa5c3d7dc6a4314617c6997c`.
No embedding was generated during download.

DATA-13A then completed a read-only audit at commit `e414d74`. It found
149,349 ZIP records, including 149,268 WAV files and 42 JSONL files, with
28,422,366,590 uncompressed member bytes. All six subsets and four acoustic
conditions match their published counts; all path-safety checks are zero.
The actual archive places audio-query metadata at each subset root and qrels
at `qrels/test.jsonl`. The small audit evidence is fixed in
`results/audits/squtr_data13a_archive_audit_20260726.json`.

Commit `fa9322b` adds DATA-13B. It performs resumable non-overwriting
extraction, full member CRC checks, strict JSONL schema validation,
corpus/query/qrels ID closure, exact condition/audio-set checks, and WAV
decode probes before creating the evaluation manifest. The public loader's
document unit and construction are explicitly recorded as one corpus row,
using `title + newline + text` when title is nonempty and `text` otherwise.
Query-text mismatches are reported because SQuTR documents an upstream
normalization step; no text is silently changed to improve results.

The first real DATA-13B attempt, `data13b_squtr_validation_20260726_141640`,
returned extraction/validation/wrapper exit codes `0/1/1`. Extraction is
complete and CRC-clean for all 149,310 files, but validation stopped at
`en/fiqa/corpus.jsonl:742` because the local validator rejected a row with
empty title and text. No final evaluation manifest was created. The pinned
official SQuTR loader preserves such a row as an empty constructed string, so
commit `c3c2c6f` removes only the stricter local rejection and adds explicit
empty-row count/example/ID-hash auditing. It does not filter the document,
invent text, alter qrels, or change the frozen candidate set. The failed
attempt remains separately recorded; a successful recovery run is still
required before any SQuTR embedding.

Commit `0212431` adds a recovery-only wrapper that validates the exact
full-CRC completion marker without rereading the already verified 49 GB of
archive/extracted content. It checkpoints audio probe metadata every 1,000
files in a cache bound to the extraction marker, structure, probe protocol,
root path, and Git commit. Interrupted content validation can therefore reuse
verified probes while mismatched cache identity is rejected.

The first recovery attempt, `data13c_squtr_content_recovery_20260726_155758`
at `68aa615`, returned reuse/validation/wrapper codes `0/1/1`. Exact extraction
reuse was verified without rereading the full tree, but content validation
stopped at the first FiQA qrels row because the local validator required a
numeric JSON score. The pinned SQuTR loader instead applies `int(...)`; the
immutable pinned FiQA reference stores all 1,706 test scores as the string
`"1"`. The repair accepts only losslessly integer-valued numbers/strings and
records actual SQuTR qrels hashes, raw types, coercion counts, and normalized
score distributions. It does not rewrite qrels or alter relevance. The
identity-only v1 probe cache and failed candidate remain preserved; the next
run uses a new v2 cache.

The next recovery invocation,
`data13c_squtr_content_recovery_20260726_161849`, did not reach extraction
reuse or validation. Its wrapper exited with code `20` because the exclusive
`flock` on `.data13c_content_recovery.lock` could not be acquired. The
intended process was no longer present when inspected, no v2 probe cache or
final manifest existed, and the lock owner had not yet been identified.
The lock file must not be deleted: file presence is not proof of a live owner,
and deleting it could split mutual exclusion across inodes. This failure is
preserved in
`results/audits/squtr_data13c_attempt3_lock_failure_20260726.json`.

The dataset-agnostic evaluator consumes explicit query/document metadata and
graded JSONL qrels, hashes frozen text embeddings and metadata before and
after scoring, and writes deterministic rankings plus per-query evidence.
OEA-5 remains `IN_PROGRESS`: candidate counts, actual qrels score
distributions, and all 149,268 audio records must first pass DATA-13C on the
remote CPU server. Only then may the official `OEA-Qwen3B (+Cl)` checkpoint
enter a small zero-shot A2T GPU smoke test.

For the ASR-uncertainty reranking main experiment, the required scope is
smaller and fixed before any result is observed: `en/fiqa` (648 queries) and
`en/nq` (3,452 queries), each under four acoustic conditions, for 16,400 audio
instances. Commit `f6f799d` adds DATA-13D with a separate manifest, resumable
probe cache, and subset-scoped lock. It preserves the failed six-subset
DATA-13C lock as evidence and never deletes, overwrites, or acquires it. The
same commit fixes two integration defects found during the audit: validator
rows now contain the downstream-required `record_id`, and FiQA identity checks
accept the official relative-path label `en/fiqa`. Thus the stale global DPC
lock no longer blocks the FiQA/NQ main experiment, while the full six-subset
OEA-5 extension remains pending.

DATA-13D then completed remotely at clean execution commit `730e0fc` in run
`data13d_squtr_fiqa_nq_validation_20260726_224653`. Extraction reuse, content
validation, FiQA identity audit, and wrapper exit codes were all zero.
Exactly 16,400 audio instances passed the decoder probe and the generated
manifest SHA256 is
`e5053d30623e5a8dd1fcf695660a0b3f8d85bd33cf5979c82b7263c2f8815e93`.
FiQA observed corpus/train/dev/test counts are 57,638/5,500/500/648; all four
SQuTR conditions contain the same 648 test IDs, and no ID or normalized-text
overlap was found between train, dev, and test. The official FiQA candidate set
contains 38 fully empty constructed documents; these rows remain in the frozen
index because the pinned loader preserves them. NQ contains 2,681,468 corpus
rows, 3,452 queries, and 4,201 qrel pairs, with exact query/corpus closure.

The source audio is not uniform: 12,300 files are 16 kHz and 4,100 are 24 kHz,
all mono. This is recorded as an observed dataset property. Every downstream
OEA or Whisper cache must therefore bind and record explicit resampling rather
than assume a single source sampling rate.

The first strict offline D2--D4 model audit ran at execution commit `92746e5`
in `asrur_d2_d4_model_audit_20260727_132612`. D3
(`BAAI/bge-base-en-v1.5`) and D4 (`BAAI/bge-reranker-v2-m3`) passed their
pinned Git revision, selected-file size, SHA256, and Git-LFS checks. D2
(`openai/whisper-large-v3`) was at the correct pinned revision and its small
files passed, but the 3,087,130,976-byte `model.safetensors` file was absent.
The audit returned `1/1` without changing or deleting any model file. At that
time, Phase 2 remained blocked on this single D2 file; D3 and D4 did not need
to be downloaded again. Compact failure evidence is retained in
`results/audits/asrur_d2_d4_model_audit_failure_20260727.json`.

The D2 repair subsequently completed in
`asrur_d2_whisper_repair_20260727_134626`. The repaired
`model.safetensors` is exactly 3,087,130,976 bytes with SHA256
`a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`.
The automatic strict offline D2--D4 audit and outer wrapper both returned zero,
with 5,823,662,271 expected and observed selected bytes and no reported errors.
This successful state supersedes the earlier failed attempt while preserving
the failure evidence. The compact success record is
`results/audits/asrur_d2_d4_model_audit_success_20260727.json`.

## OEA-6 measurement boundary

Existing full Clotho embedding runs record approximately 9.31 GiB allocated
and 10.12 GiB reserved on an RTX 4090. These are workload-level peak-memory
observations, not yet a controlled per-query latency/throughput benchmark.
Hardware-specific results will be reported separately from paper values.

The formal implementation now pins the paper values (539.3 ms/audio, 2.60
ms/text, 11.6 GB, and 16.2M parameters) and a separate reproducible measurement
protocol: one A100-SXM4-80GB, BF16, batch size 1, ten warmup calls per modality,
the complete Clotho evaluation split, per-call CUDA synchronization, and
end-to-end public-encoder wall time. It records every raw latency and reports
mean, population standard deviation, P50, P95, throughput, model-resident and
peak memory, load time, and LoRA/projection parameter counts. Because the paper
does not publish these timing details, reproduced measurements will remain
labelled `[INFERRED]` rather than silently treated as the exact paper protocol.

The first attempted efficiency run, at commit `931a71f`, deliberately retained
its failure evidence after the A100 configuration was scheduled on an RTX
4090. The model, checkpoint, manifest, and clean-worktree checks passed and the
model loaded successfully; the exact GPU-name guard then stopped the run before
warmup or latency measurement. This is classified as an expected scheduling
guard rather than a model failure. Its small evidence is retained in
`results/audits/qwen3b_clotho_a100_efficiency_on_rtx4090_failed_20260721.json`.

Commit `4676525` adds a separate RTX 4090 entrypoint with the identical model,
data order, warmup, batch size, timer, synchronization, and timing boundary.
It has a distinct config and experiment prefix, requires the exact
`NVIDIA GeForce RTX 4090` name, and writes an explicit `[INFERRED]
hardware-mismatched` claim scope into `metrics.json`. That result may be used
as the same-4090 SpeechXBT baseline and displayed beside the paper reference,
but it cannot establish whether the A100 paper latency or peak memory was
reproduced. The original A100 entrypoint and guard remain intact.

The RTX 4090 run completed at commit `a2c13da` with exit code 0 and a clean
worktree. It measured all 1,045 audio clips and all 5,225 captions. Audio
encoding was 295.515 ms mean (83.645 ms population standard deviation),
287.715 ms P50, 423.286 ms P95, and 3.384 clips/s. Text encoding was 38.373 ms
mean (1.631 ms population standard deviation), 38.228 ms P50, 40.688 ms P95,
and 26.060 queries/s. Peak allocated/reserved memory was 9.307/10.115 GiB.
The raw 6,270 latency rows have SHA256 `1e66e0cb...7efc2`, and the finalized
metrics SHA256 is `fa3c277e...41605`.

These latency and memory values are not assigned a Table 5/16 reproduction
status: the GPU differs from PAPER and PAPER omits the exact timing scope. The
large text-latency difference also demonstrates why the end-to-end timing
boundary must remain explicit instead of being silently compared with a
possibly GPU-only or batched paper measurement.

The released checkpoint contains 12,615,680 LoRA parameters and two 1,049,600
parameter projection heads, for an exact measured total of 14.71488M. PAPER
reports 16.2M but does not publish its counting procedure or a different
checkpoint structure. The 1.48512M (9.167%) shortfall is therefore recorded as
an unresolved `[PAPER]` versus `[CODE+MEASURED]` discrepancy, not filled by an
inferred component. Full small evidence is in
`results/audits/qwen3b_clotho_rtx4090_efficiency_20260721.json`.

The A100 run then completed at commit `9f62e20` on exactly one
`NVIDIA A100-SXM4-80GB`, with BF16 available, exit code 0, an empty final Git
status, and no active GPU process after completion. All 1,045 audio clips and
5,225 captions were measured. Audio encoding was 273.207 ms mean (65.855 ms
population standard deviation), 267.578 ms P50, 381.090 ms P95, and 3.660
clips/s. Text encoding was 46.638 ms mean (4.225 ms population standard
deviation), 45.318 ms P50, 52.117 ms P95, and 21.442 queries/s. Peak
allocated/reserved memory was 9.307/10.115 GiB (9.994/10.861 decimal GB).
The raw 6,270 latency rows have SHA256 `4e4e4003...6a5ae`; the finalized
metrics SHA256 is `25702e36...b6095`.

This is a completed `[INFERRED] same-hardware-model` controlled benchmark, but
not an exact paper-protocol reproduction. Relative to PAPER, its declared
end-to-end means are 49.340% lower for audio and 1,693.774% higher for text.
Those large and opposing deltas cannot be attributed to the GPU or model:
PAPER omits its timer, batching, synchronization, preprocessing, cache, and
device-transfer boundary. PAPER also does not define allocated versus reserved
peak memory or decimal GB versus GiB. Consequently the eight strict Table 5
and duplicate Table 16 observations are recorded as `blocked`, while the
controlled numeric measurements remain visible in the audit evidence.

The same released checkpoint again contains exactly 14.71488M LoRA plus
projection-head parameters, 1.48512M (9.167%) below PAPER's 16.2M. The public
materials do not provide the training-time counting procedure or identify the
checkpoint variant used for the efficiency row, so no missing component is
invented. Complete A100 evidence is retained in
`results/audits/qwen3b_clotho_a100_efficiency_20260721.json`.

## ASRUR Phase 1: OEA-Nemo3B (+Cl) Clotho T2A

The Phase 1 correctness check used the pinned
`JudeJiwoo/OEA-Nemo3B-Cl` revision
`9588912298afca0b11f5895b864ae28083f35022`, its checksum-verified
59,072,047-byte inference-only checkpoint, and the canonical model-lock SHA256
`fd09e4d2...c8c2`. A strict-offline RTX 4090 run generated all 1,045 audio
embeddings and 5,225 caption embeddings with no pending chunks. The resulting
arrays have shapes 1,045×512 and 5,225×512; peak allocated/reserved memory was
9.39/10.48 GiB. The generation run and its two non-fatal Transformers
compatibility warnings are recorded in
`results/audits/asrur_nemo_g1_full_embeddings_20260727.json`.

The subsequent CPU-only retrieval suite completed with empty stderr and zero
suite, attempt, and outer return codes. PAPER reports Clotho T2A R@1/5/10 of
21.57/47.16/60.36 for OEA-Nemo3B (+Cl). The two protocols declared before
evaluation produced:

| Protocol | Queries | R@1 | R@5 | R@10 | Maximum absolute delta |
|---|---:|---:|---:|---:|---:|
| `[CODE]` default joint, all captions | 5,225 | 21.7225 | 47.1196 | 60.4402 | 0.1525 pp |
| `[CODE]` T2A-only, seed0 one caption | 1,045 | 21.6268 | 46.7943 | 59.8086 | 0.5514 pp |

Both predeclared public-code protocols are within one percentage point on every
reported recall and therefore pass the Phase 1 numerical correctness gate. The
paper does not disclose which caption-selection protocol produced its row, so
neither observed protocol is selected post hoc as the strict paper protocol.
The six numeric observations are recorded as `close`; the three strict-paper
observations remain `blocked`. The public runtime also omits the audio
`passage:` prefix stated by PAPER, and this conflict remains explicit. Complete
small evidence is in
`results/audits/asrur_nemo_clotho_t2a_20260727.json`; large rankings remain in
the remote suite directory.

## ASRUR Phase 1 direction gate: OEA-Nemo3B (+Cl) Clotho A2T

The same immutable Nemo embedding run was reused on CPU for the direction
required by the proposed spoken-query task: 1,045 audio queries were ranked
against 5,225 frozen caption candidates, with all five captions belonging to
the matching clip treated as positives. The run completed with empty stderr
and produced R@1/5/10 of 26.88995/51.38756/65.26316, MRR of 0.38762, and DCG
of 0.51023. This passes the direction sanity gate, but it is explicitly a
`[CODE]` extension rather than a paper-table reproduction. Compact evidence is
stored in `results/audits/asrur_nemo_clotho_a2t_20260727.json`; full rankings
remain in the recorded remote suite directory.

The original `nvidia/omni-embed-nemotron-3b` snapshot was independently
audited on CPU before Phase 2. Its remote portable lock and tracked canonical
lock are byte-for-byte identical: 6,416 bytes with SHA256
`eb62da7579d2f5b7ccd774e15d518d3ed50f4d502753b3fc4a7f51187b96cc1c`.
The lock intentionally claims only the immutable base snapshot and public-code
pooling/prompt protocol; it is not itself a reproduced retrieval result.

Phase-2 CPU dry-run attempt 1 stopped after OEA config resolution because the
vanilla config builder's direct script entrypoint could not import the
repository-level `scripts` package. The exact error was
`ModuleNotFoundError: No module named 'scripts'`. No model was loaded and no
metric was calculated. The minimal fix adds the repository root to `sys.path`
only for direct script execution, without changing protocol, pooling, model
loading, scores, or metrics. A regression test now exercises the exact
entrypoint mode, and all 388 repository tests pass. The failed attempt remains
recorded in `results/audits/asrur_phase2_dry_run_import_failure_20260727.json`.

After the minimal direct-entrypoint fix, Phase-2 CPU dry-run attempt 2
completed at execution commit `1ff9fc06e16f45967965937d4c001e58661b2b6f`.
All 17 planned steps exited zero; the run, wrapper, and caller exit codes were
also zero, and stderr was empty. The dry-run validated the four-condition
OEA/vanilla/Whisper plan, both BGE dev templates, the 57,638-document FiQA
corpus, 500 dev queries, and 648 test audio queries per acoustic condition.
It loaded no model, used no GPU or network, and produced no research metric.
Compact evidence is stored in
`results/audits/asrur_phase2_dry_run_success_20260727.json`. The remaining
Phase-2 gate is a separately approved one-RTX-4090 formal execution.

Formal Phase-2 attempt 2 at execution commit
`ee637721c1f7f6ef13f63addfe13c0b4d172f9f2` completed and published all four
OEA embedding caches, all four vanilla-Nemotron caches, and the BGE
corpus/two-dev-template caches, but then failed in `whisper_clean`. All 648
records raised the same exception: the feature extractor emitted FP32
`input_features` while the frozen Whisper model's first convolution had BF16
weights and bias. This is a model-adapter dtype bug, not OOM or corrupt audio;
no completion manifest or formal metric was produced. Repair commit
`5835fab272ac0e28a081d5850cf420bbe2bfca18` casts only floating model inputs
to the loaded model dtype, preserves mask dtypes, sets the score-return
generation options coherently, fails closed if required beam scores are
absent, and stops after eight consecutive failures. It also permits only
schema/size/SHA256-verified read-only reuse of the 11 complete cache groups
from the failed commit. The repair passes 407 repository tests; a one-record
real RTX 4090 Whisper smoke remains required before attempt 3. Compact failure
evidence is stored in
`results/audits/asrur_phase2_whisper_dtype_failure_20260727.json`.

The first one-record repair smoke at execution commit `6ef3c2c` confirmed that
the input-dtype failure was removed and that all pinned D2--D4 resources passed
strict local verification. It then exposed a second, independent compatibility
failure in the Transformers 4.52.4 Whisper generation wrapper: the wrapper
expands `num_return_sequences`, re-stacks beam score rows, and retains global
beam indices, after which transition-score gathering indexes beyond the
compressed tensor. CUDA consequently raised a `ScatterGatherKernel`
out-of-bounds assertion for the selected record. The minimal repair keeps the
same frozen checkpoint and four-best proxy-posterior protocol, constructs the
English/transcribe/no-timestamps decoder prompt explicitly, and calls the
model through the base `GenerationMixin` beam search. It does not change the
data, candidate set, or evaluation. The repair passes 15 targeted and 409
repository tests; a second one-record GPU smoke is required before Phase-2
attempt 3. Compact evidence is stored in
`results/audits/asrur_whisper_beam_index_smoke_failure_20260727.json`.

The second one-record smoke at execution commit `ef3a77a` completed the base
four-beam generation path, confirming that neither the dtype mismatch nor the
beam-index CUDA gather failure recurred. It then failed closed because a
generation transition score aligned with a retained, non-special token was
non-finite. No N-best artifact or metric was emitted. The implementation does
not discard or clamp that value. Instead, the same frozen model now scores the
exact four generated sequences with a teacher-forced forward pass and
float32 cross entropy, excluding prompt and special tokens. Beam
`sequence_score` remains separate; the resulting average token log
probability remains explicitly an uncalibrated proxy. This scoring method is
part of the immutable cache identity, and per-record failures now carry an
exact stage. Compact evidence is stored in
`results/audits/asrur_whisper_transition_score_smoke_failure_20260727.json`.

The third one-record smoke at execution commit `54d0e72` passed the complete
strict-offline gate in 48 seconds. The same `en/fiqa:clean:4641` record yielded
exactly four hypotheses; all four beam sequence scores and teacher-forced
proxy scores were finite, and generation transition scores were not used.
The wrapper exited zero, stderr was empty, and no failure artifact was
produced. The 923-byte N-best artifact has SHA256
`fe42d5fd20a49185dc32ba52cd5332cbc6be3eee42ad32c733aed24970bb5385`.
This is a correctness gate rather than a research metric. It authorizes
Phase-2 attempt 3 to reuse the 11 checksummed complete OEA, vanilla-Nemotron,
and BGE cache groups from attempt 2 while generating new four-condition
Whisper caches. Compact evidence is stored in
`results/audits/asrur_whisper_teacher_forced_smoke_success_20260728.json`.

Phase-2 attempt 3 subsequently completed the clean, 20 dB, and 10 dB Whisper
caches and 647 of 648 zero-dB shards. The only failed record was
`en/fiqa:snr_0:10639`. Its 3.64-second, 16 kHz mono PCM audio exists and is
decodable; four-beam generation completed, but N-best validation found a
non-finite teacher-forced conditional log-probability for at least one
retained non-special token. The wrapper failed closed, preserved every
completed shard, released the GPU, and emitted no completion manifest or
formal metric. The implementation does not filter, clamp, or replace the
score. An isolated diagnostic now compares batched BF16, per-hypothesis BF16,
and FP32-upcast scoring of the exact same BF16-generated beam sequences before
a repair is selected. Compact failure evidence is stored in
`results/audits/asrur_phase2_attempt3_single_record_failure_20260728.json`.

The isolated retry at execution commit `c294730` completed in 33 seconds on
one RTX 4090. For the same `en/fiqa:snr_0:10639` record, all four beam sequence
scores were finite, and every retained non-special token score was finite in
BF16 batched, BF16 per-hypothesis, and FP32-upcast exact-sequence scoring.
Peak allocated GPU memory was 6,448,808,960 bytes. The attempt-3 failure is
therefore not a stable property of the audio or one fixed scoring path. The
evidence is consistent with a transient same-protocol low-precision numerical
failure, but it does not identify a lower-level CUDA kernel root cause.

The recovery policy remains fail-closed. Complete caches are reused only
after their immutable outputs verify. Partial reuse validates every shard
against the expected query, condition, audio path, four-hypothesis schema, and
finite scores, and records the exact source Git commit and per-file hashes.
Only explicitly classified non-finite Whisper failures may be retried, at
most three times, without changing the model, dtype, decoding, scoring, or
input. Every failed attempt is retained, and only the first strict finite
same-protocol result may become a shard. Filtering, clamping, replacing a
score, or silently switching to FP32 is prohibited. Success evidence is
stored in
`results/audits/asrur_whisper_numeric_diagnostic_success_20260728.json`.

Phase-2 attempt 4 then completed at execution commit `d9baf22`. The wrapper
and all 44 recorded steps exited zero. Fourteen immutable complete caches were
verified and reused, 647 `snr_0` Whisper shards were validated and copied with
source-commit provenance, and the missing record succeeded on its first new
attempt. The final Whisper condition contains 648 shards, a complete cache
manifest, and no new failure record.

The successful execution produced a scientific NO-GO, not a usable candidate
pipeline. On Clean SQuTR-FiQA, the original Omni backbone reached
nDCG@10=`0.258495` and Recall@100=`0.602574`, while Gold+BGE reached
nDCG@10=`0.405853`. These non-trivial controls argue against a global
audio/corpus/qrels identity failure. In contrast, the selected
OEA-Nemo3B(+Cl) route reached nDCG@10=`0` and Recall@100=`0.002561`, and its
Top-100 oracle nDCG@10 was only `0.003761`. Whisper 1-best+BGE also reached
nDCG@10=`0`. Every preregistered check failed under all four conditions, so
the overall decision is `NO_GO_REQUIRES_USER_DECISION`; changing checkpoint,
ASR generation, or candidate generation is not automatically authorized.

The compact execution evidence is
`results/audits/asrur_phase2_attempt4_no_go_20260728.json`. A CPU-only
diagnostic now measures cached Whisper WER/content/diversity and compares OEA
versus vanilla embedding geometry and retrieval hubness. Phase 3, CE scoring,
fusion, and gate training remain stopped until the diagnostic separates
implementation/protocol errors from genuine cross-domain model failure.

The completed CPU diagnostic established that the formal Whisper cache is
invalid rather than merely weak. Clean, 20 dB, and 10 dB each contain one
unique Top-1 transcript across all 648 queries: `Thank you.`; whitespace
case-folded corpus WER is `1.0`. At 0 dB, 647/648 records have the same
hallucination. The selected OEA projected space also shows severe anisotropy:
on Clean, sampled audio/audio and document/document mean off-diagonal cosine
similarities are `0.755988` and `0.509644`, compared with `0.386363` and
`0.196054` for the original Omni control. OEA Top-100 includes a positive for
only 6/648 Clean queries, versus 498/648 for vanilla. These observations do not
authorize changing the formal ASR or candidate protocol. A small three-way
GPU diagnostic now compares the current generic BF16 four-best entrypoint
against official Whisper BF16 and FP32 one-best on identical audio.

That differential completed at `576c4f6` on eight fixed records. Current
generic BF16 four-best produced the intended FiQA content with corpus
WER=`0.141304`; official BF16 and FP32 one-best were identical and each
reached WER=`0.130435`. The pinned model, present audio, BF16 precision, and
current generic generation entrypoint therefore pass the fixed sample. This
does not retroactively validate the attempt-4 cache or identify a unique
historical trigger from its completed artifacts. Attempt 5 consequently
preserves that cache as failure evidence, reuses only the 11 complete
OEA/vanilla/BGE groups, regenerates every Whisper record in a new cache root,
and applies a per-condition catastrophic-content integrity gate before any
ASR ranking is accepted.

The OEA-Nemo3B(+Cl) failure was separately localized at execution commit
`dc995b5`. On eight fixed Clean FiQA queries, all relevant positive documents,
and 64 fixed negatives, the frozen base hidden space remained strongly
retrievable (R@1=`0.875`, R@10=`1.0`). Applying only the released OEA LoRA
reduced R@10 to `0.75` and raised mean audio/audio cosine from `0.372756` to
`0.708041`. Applying only the OEA modality heads to base hidden states reduced
R@10 to `0.5`. Applying both released components was worst: R@1=`0`,
R@10=`0.25`, mean audio/audio cosine=`0.843654`, and mean positive rank=`23`.
Fresh full audio embeddings matched the prior OEA cache exactly
(mean/minimum row cosine=`1.0`), while fresh base audio matched the vanilla
cache at mean cosine `0.999999`. The global collapse is therefore not explained
by stale audio embeddings, the loader selecting a different checkpoint, or a
globally broken FiQA/SQuTR identity.

The evidence supports, but does not by itself causally prove, a transfer/domain
failure in the retrieval-specific LoRA and modality heads. This interpretation
is consistent with the paper's stated curriculum: WavCaps audio captions,
AudioCaps caption retrieval, and an optional final 3,839-clip Clotho stage that
is specifically described as improving natural audio descriptions
(`Omni-Embed-Audio.pdf`, page 5). The paper also warns that AudioCaps and
Clotho caption queries mirror training distributions (pages 1 and 3). Spoken
financial questions paired with long FiQA documents are outside those reported
training and evaluation distributions. The checkpoint is not globally corrupt:
the same locked `+Cl` weights already reproduced Clotho T2A and passed the
Clotho A2T direction check.

One remaining implementation question is localized text-cache disagreement:
the eight-query diagnostic found full-text fresh/cache mean cosine `0.995951`
but a minimum of `0.663597`. This is too sparse to explain the global
audio-space collapse, but it must not be ignored. The independent full rerun
therefore re-encodes all 57,638 FiQA documents and all 2,592 four-condition
audio queries under the same checkpoint, public-code audio protocol, text
prefix, BF16 setting, and exact Top-100 evaluator. It reuses no previous OEA
embedding or symlink, preserves partial chunks for recovery, and compares every
fresh row and metric with attempt 4. It does not change or select a checkpoint,
train a model, download data, or alter candidate generation. The runner and
audit are `scripts/run_asrur_oea_cl_fiqa_rerun.sh` and
`scripts/audit_asrur_oea_cl_fiqa_rerun.py`.

Phase-2 attempt 5 subsequently completed at execution commit `dc995b5` on
host `bitahub-a20626401511337984908046`. The wrapper exited zero and the
completion manifest is `complete`. All 2,592 Whisper records were regenerated
in a new cache root; no attempt-4 Whisper shard was reused. Every acoustic
condition passed the preregistered content gate with 648 unique Top-1
transcripts, mode fraction `0.001543`, and no violation. Corpus WER for
Clean/20/10/0 dB is `0.127663/0.127663/0.125123/0.147411`.

The repaired Whisper 1-best+BGE baseline is both strong and stable:
nDCG@10 for Clean/20/10/0 dB is
`0.392020/0.392082/0.391101/0.379271`, compared with the fixed Gold+BGE
upper bound `0.405853`. The Clean-to-0 dB absolute nDCG@10 decrease is
`0.012749` (relative `3.2521%`). It also exceeds the original Omni baseline
by `0.133525/0.133634/0.141325/0.165228` nDCG@10 across the four conditions.
This closes ASR-E1 and invalidates the earlier conclusion that Whisper itself
was unusable; the attempt-4 Whisper output remains preserved only as failure
evidence.

The scientific Go/No-Go decision nevertheless remains NO-GO for the proposed
OEA-Top-100 reranking pipeline. Attempt 5 did not regenerate OEA artifacts,
and the fixed OEA candidate Recall@100 remains
`0.002561/0.001941/0.001337/0.001646`; its oracle nDCG@10 remains
`0.003761/0.002734/0.001972/0.002302`. A cross-encoder or learned gate cannot
recover relevant documents that are absent from the candidate set. The
separately approved, same-checkpoint, same-protocol, zero-old-OEA-cache full
rerun is therefore still the next decision gate. Compact attempt-5 evidence is
stored in `results/audits/asrur_phase2_attempt5_success_20260728.json`.
