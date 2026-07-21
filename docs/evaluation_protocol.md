# Canonical retrieval evaluation protocol

This document separates paper-stated choices from public-code behavior and
unpublished protocol details. It must be updated before any result is labelled
as a reproduced paper value.

## Confirmed scope

- `[PAPER]` Table 2 evaluates text-to-audio (T2A) retrieval on AudioCaps (975
  clips, five captions each), Clotho (1,045 clips, five captions each), and
  MECAT (847 auto-captioned pairs), reporting R@1, R@5, and R@10.
- `[PAPER]` Table 3 evaluates caption-based text-to-text (T2T) retrieval on the
  same datasets and reports R@1, R@5, and R@10.
- `[PAPER]` Positive UIQ types use query-to-audio R@1, R@5, and R@10.
- `[CODE]` The public implementation computes rank as one plus the number of
  candidate scores strictly greater than the target score. This is an
  optimistic tie policy.

## Unpublished or conflicting details

- `[MISSING]` The paper does not state whether T2T uses all five captions as
  queries or samples one caption per clip.
- `[CODE]` `eval_core.evaluate` defaults to one randomly selected T2T caption
  per clip. The random generator uses seed 0 unless a sample seed is supplied.
- `[CODE]` T2A behavior depends on which tasks are requested: a T2A-only run
  samples one caption per clip, while a joint T2A/A2T run evaluates all
  captions. This makes the published command path ambiguous.
- `[MISSING]` The paper does not state a tie policy.
- `[MISSING]` The exact MECAT 847-candidate manifest is not released; the UIQ
  files contain 848 positive IDs.

## Canonical implementation rules

`AudioRetrieval.evaluation.canonical` is an embedding-only layer with no model,
dataset, or random-number side effects.

1. Candidate IDs must be unique.
2. Every query must resolve to at least one candidate; missing targets are
   errors and are never silently skipped.
3. Query subsets are explicit ordered indices. No caption sampling occurs
   inside the evaluator.
4. T2A and positive UIQ use one exact target audio ID per query.
5. Caption T2T excludes the query caption itself and treats every other
   caption from the same clip as positive.
6. Embeddings are L2-normalized before cosine similarity; zero-norm, NaN, and
   infinite rows are errors.
7. Metric rank uses the public code's optimistic strict-greater policy and the
   report records `tie_policy=optimistic_strict_greater`.
8. Full candidate rankings use descending score with candidate index as a
   deterministic tie breaker.
9. Formal paper-table runs must save the explicit query indices, candidate
   IDs, positive indices, embeddings, rankings, metrics, and Git commit.

### Audio-to-text extension for SpeechXBT-OEA

`[CODE]` The public baseline and precomputed-embedding runners also implement
audio-to-text (A2T), although the paper does not report an A2T table. For
Clotho, every one of the 1,045 audio embeddings is a query, all 5,225 caption
embeddings form the frozen text candidate bank, and all five captions with the
same exact clip ID are positives. Rank uses the best-scoring positive under
the same optimistic strict-greater tie policy. Missing target groups, duplicate
audio query IDs, count drift, and implicit query filtering are errors.

The fixed official-checkpoint entrypoint is:

```bash
bash scripts/run_qwen3b_cl_clotho_a2t_suite.sh \
  /absolute/path/to/completed_qwen3b_cl_embedding_directory
```

This CPU-only suite reuses the already generated, lock-bound Qwen3B-Cl Clotho
embeddings. It writes all similarities, rankings, multi-positive mappings,
input/output SHA256 values, Git state, environment, commands, logs, and final
R@1/5/10, MRR, and DCG. Its `paper_table` is explicitly
`OEA-4 extension (not paper-reported)`; it must not be presented as a missing
OEA paper value.

### Frozen target-corpus index contract

`scripts/evaluate_frozen_text_index.py` is the dataset-agnostic OEA-5
evaluator. It requires unique explicit query/document IDs and an external
JSONL qrels file; missing queries, missing documents, duplicate qrels, and
non-positive relevance grades are hard errors. It reports R@1/5/10, MRR@10,
and graded nDCG@10 with deterministic candidate-index tie breaking.

The candidate embedding and metadata files are the frozen text index. Their
byte sizes and SHA256 values are checked before and after scoring; a change
during evaluation fails the run. Large index files are referenced by identity
rather than duplicated in every result directory. The result retains the
index identity, top ranking indices/scores, per-query evidence, qrels/input
hashes, Git state, command, logs, and failure traceback. The runnable wrapper
is:

```bash
bash scripts/run_frozen_text_index_evaluation.sh \
  /absolute/path/to/fixed_config.json \
  /absolute/path/to/new_output_directory
```

`configs/eval/frozen_text_index_example.json` is intentionally non-runnable:
the target corpus, split, document construction, qrels semantics, and absolute
artifact paths remain `[MISSING]` until they are explicitly frozen.

### OEA-Qwen3B efficiency benchmark

`[PAPER]` Table 5 and Appendix Table 16 report OEA-Qwen3B on Clotho and an
A100-SXM4-80GB as 539.3 ms/audio clip, 2.60 ms/text query, 11.6 GB peak GPU
memory, and 16.2M trainable parameters. `[MISSING]` The paper does not publish
warmup count, repeat count, batch size, timer implementation, cache state, or
the exact preprocessing/device-transfer timing boundary.

The committed `[INFERRED]` reproduction protocol therefore fixes one visible
`NVIDIA A100-SXM4-80GB`, BF16, batch size 1, ten warmup audio and text calls,
all 1,045 Clotho audio clips and all 5,225 captions in canonical manifest
order, `time.perf_counter_ns`, and a CUDA synchronization immediately before
and after every public `encode_batch` call. The wall-clock scope includes
preprocessing, model forward, projection, L2 normalization, and device-to-host
copy. Results report raw per-item latency plus mean, population standard
deviation, min, P50, P95, max, throughput, model-resident memory, workload peak
allocated/reserved memory, model-load time, and LoRA/head parameter counts.

The formal non-resumable entrypoint is:

```bash
bash scripts/run_qwen3b_cl_clotho_efficiency.sh
```

The wrapper resolves the committed Qwen3B-Cl model lock, requires a clean
worktree, runs fully offline, and refuses non-A100 hardware for the
paper-comparison result. Measurements on an RTX 4090 or other GPU must use a
separate experiment label and must not be compared as an exact absolute-latency
reproduction.

The RTX 4090 engineering comparison uses the identical checkpoint, manifest,
sample order, warmup, batch size, timer, synchronization, and timing boundary,
but a distinct immutable hardware requirement and experiment prefix:

```bash
bash scripts/run_qwen3b_cl_clotho_efficiency_rtx4090.sh
```

Its config is
`configs/eval/qwen3b_cl_clotho_efficiency_rtx4090.json`. The result is labelled
`[INFERRED] hardware-mismatched`: it may be shown beside the paper values and
used as the same-RTX4090 SpeechXBT baseline, but its latency, throughput, and
peak-memory deltas cannot determine whether the A100 paper result was reproduced.
The A100 entrypoint and hardware guard remain unchanged.

## Required protocol resolution

Before Table 2 or Table 3 is called an exact reproduction, obtain the authors'
query-selection and MECAT manifest details or run clearly separated protocol
variants. Until then, results must be labelled `[CODE]` protocol or
`[INFERRED]` protocol rather than `[PAPER]` protocol.

## Released UIQ schema audit

- `[CODE]` All 15 released JSONL files use
  `audio_id/dataset/dataset_slug/query_type`, not the legacy
  `clip_id/uiq[].bucket/query` schema consumed by the public `UIQRunner`.
- `[CODE]` Positive rows store their text in `generated_query`; negative rows
  store it in `negative_query` and include `negative_captions`.
- `[CODE]` The release contains exactly 13,053 rows: 4,530 AudioCaps, 4,722
  Clotho, and 3,801 MECAT rows.
- `[CODE]` Each positive file contains one row per audio ID. Negative files
  intentionally repeat targets: 630/542/409 rows correspond to only
  255/247/176 unique AudioCaps/Clotho/MECAT target IDs.
- `[CODE]` Negative rows do not contain a hard-negative audio ID despite the
  UIQ README saying they reference a pre-mined clip. They must not be used for
  HNSR/TFR-HN until a pairing is obtained or explicitly reconstructed.
- `[CODE]` Clotho positive IDs include `.wav`, while negative IDs are stems.
  The adapter preserves these IDs verbatim; candidate resolution must record
  any exact or stem-based mapping rather than silently normalizing them.

`AudioRetrieval.evaluation.uiq_schema` validates this released schema, keeps
duplicate negative rows, rejects duplicate positive IDs, and exposes the query
text through a common immutable record. It does not invent missing HN IDs.

## Embedding evaluation artifact contract

`scripts/evaluate_embedding_artifacts.py` evaluates already generated `.npy`
embeddings without loading a model. `scripts/run_embedding_evaluation.sh`
provides the remote CPU wrapper and captures stdout/stderr.

Each run uses a new directory whose basename equals `experiment_id`. Existing
artifacts are never overwritten. A successful formal run contains at least:

- `config.yaml`, `command.sh`, `python_command.sh`, `environment.txt`,
  `python_environment.txt`, `gpu_info.txt`, `git_commit.txt`, and
  `git_status.txt`;
- copied `query_embeddings.npy` and `candidate_embeddings.npy` plus their
  JSONL metadata;
- `similarities.npy`, `ranks.npy`, `rankings.npy`, explicit positive/ignored
  indices, and evaluated query indices;
- `metrics.json` with protocol label/source, task, paper table, input and
  artifact SHA256 values, shapes, counts, tie policy, and R@1/5/10;
- wrapper `stdout.log`, `stderr.log`, and `exit_code.txt`.

Formal evaluation rejects a dirty Git worktree by default. T2T configuration
must declare `query_selection=all` or reference an immutable JSON list with
`query_selection=indices`; no random selection is performed. Failed runs keep
`metrics.json` with the exception and traceback.

The input file is JSON (and is copied as JSON-compatible `config.yaml`). The
minimum T2A/UIQ configuration is:

```json
{
  "schema_version": 1,
  "experiment_id": "oea_qwen3b_clotho_t2a_seed42_YYYYMMDD_HHMMSS",
  "model": "OEA-Qwen3B (+Cl)",
  "checkpoint": "/absolute/path/to/step_40.pt",
  "dataset": "Clotho v2 evaluation",
  "task": "t2a",
  "paper_table": "Table 2",
  "protocol_label": "canonical-all-captions",
  "protocol_source": "INFERRED",
  "seed": 42,
  "query_embeddings": "/absolute/path/to/query_embeddings.npy",
  "candidate_embeddings": "/absolute/path/to/candidate_embeddings.npy",
  "query_metadata": "/absolute/path/to/query_metadata.jsonl",
  "candidate_metadata": "/absolute/path/to/candidate_metadata.jsonl",
  "query_selection": "all"
}
```

`model`, `checkpoint`, `dataset`, and integer `seed` are mandatory. The
evaluator records the seed for provenance but does not use randomness itself.
For T2T, candidate paths may be omitted because the query caption bank is also
the candidate bank. Because the paper does not publish caption-selection
details, a chosen protocol must not be marked `PAPER` unless new evidence is
found.

## Negative-query artifact contract

The general evaluator does not guess a hard-negative audio ID.
`scripts/evaluate_negative_embedding_artifacts.py` is a separate strict path
for Figure 3 and Table 17. It requires an explicit JSONL mapping from every
query ID to one target candidate ID and one distinct hard-negative candidate
ID. The mapping must cover the complete query metadata exactly, and every
audio ID must resolve in the fixed candidate collection. Missing or duplicate
IDs are errors; `negative_captions` are never treated as audio identifiers.

The runner saves full similarities/rankings, optimistic one-based target/HN
ranks, per-query evidence, R@k, Delta-Rank, HNSR/HNSR@k, TFR/TFR-HN@k, all
source/artifact hashes, and the same Git/environment/command provenance as the
general evaluator. The full contract and JSON example are in
`docs/negative_query_metrics.md`.

`[MISSING]` The released 1,581 negative UIQ rows still lack hard-negative
audio IDs. Therefore this executable path does not make Tables 4/17
reproducible by itself; a formal run remains blocked until an author pairing
or a separately labelled, auditable reconstruction is available.

## Resumable official-checkpoint embedding generation

Before the full candidate/query bank is authorized, run the lock-bound fixture
entrypoint on one BF16-capable GPU:

```bash
bash scripts/run_qwen3b_cl_clotho_lock_bound_smoke.sh
```

This is a formal-path smoke, not a paper result. It resolves the committed
Qwen3B-Cl model lock, verifies the same complete base/checkpoint inventory used
by the full run, and invokes the same resumable generator and public-code prompt
protocol on 5 bundled Clotho audio candidates and all 25 associated captions.
All formal Qwen3B-Cl GPU wrappers fail closed unless PyTorch sees exactly one
CUDA GPU with BF16 support. GPU model and total memory are recorded rather than
restricted: the measured RTX 4090 smoke used 9.14 GiB allocated, while Table 5
efficiency measurements remain A100-SXM4-80GB-specific. The observed inventory
and any rejection reasons are saved as `gpu_preflight.json` in the attempt.
The full wrapper remains a separate long operation and must not be started until
the smoke report reaches `status=complete` with 5 candidate and 25 query rows.

`scripts/generate_oea_embeddings.py` and
`scripts/run_qwen3b_cl_clotho_embeddings.sh` implement the first formal
OEA-Qwen3B (+Cl) embedding pass over the canonical Clotho evaluation manifest.
The fixed protocol configuration is
`configs/eval/qwen3b_cl_clotho_embeddings.json`. It is not accepted directly
by a formal GPU run. The wrapper first requires the committed portable lock
`results/model_locks/oea_qwen3b_cl.json`, then invokes
`scripts/build_official_oea_eval_config.py` to create
`resolved_embedding_config.json` inside the stable experiment directory.
The resolver requires a clean worktree and both inputs to be tracked by Git;
it refuses to overwrite a different existing result and exactly verifies an
identical result on resume.

The resolved configuration replaces the protocol's resource subset with the
lock's complete base-model inventory and derived-checkpoint identity. At GPU
runtime, every locked base file (including tokenizer, processor, and model
configuration files), every weight shard, the derived checkpoint, the model
lock itself, and the source protocol are checked by byte size and SHA256. The
variant/model/revision/path mappings must agree at every layer. A lock or
protocol change, a resource drift, a different Git commit, or a manually
edited resolved configuration is a hard error. The model lock is therefore
generated on CPU, reviewed and committed as a small JSON artifact before any
formal GPU embedding run; an untracked lock is intentionally rejected.

The generator preserves manifest order and creates exactly 1,045 audio
candidates and 5,225 caption queries. Each projected 512-dimensional chunk is
written atomically, checked for finite values and unit L2 norm, and accompanied
by its range and SHA256. A resumed invocation verifies every completed chunk
before skipping it. The full arrays are consolidated only after all chunks are
present. Immutable config and metadata files are verified instead of
overwritten, and a changed Git/config/manifest identity cannot reuse an old run
directory. Every invocation has a separate `attempts/attempt_*` directory with
command, Git state, environment, GPU inventory, stdout, stderr, exit code, and
failure evidence.

The committed audio/text batch sizes are both 1 `[INFERRED]`, selected as the
conservative first formal setting after the five-sample A100 smoke test. They
affect throughput, not the candidate or query set. Any later batch-size change
requires a new committed configuration and experiment identity.

There is a material prompt conflict: `[PAPER]` states that audio uses the
`passage:` prefix, while `[CODE]` `_build_audio_messages()` explicitly omits a
text prefix from audio-only messages. The first checkpoint evaluation follows
the public-code audio-only/no-prefix path and is labelled accordingly. A
prefix-including A/B run must be reported separately; neither result may be
silently substituted for the other.

## Qwen3B-Cl Clotho retrieval suite

After the resumable generator reaches `status=complete`, the fixed CPU-only
suite is launched with:

```bash
bash scripts/run_qwen3b_clotho_retrieval_suite.sh \
  /absolute/path/to/completed_embedding_directory
```

Its source config is
`configs/eval/qwen3b_cl_clotho_retrieval_suite.json`. Before evaluating, the
suite rechecks the generation config SHA256, checkpoint revision and SHA256,
the committed protocol and model-lock identities, the resolved config source,
generator Git ancestry, generation status, every recorded artifact hash,
metadata order/counts, array shapes and finiteness, and unit L2 norms. The
prepare/finalize implementation is
`scripts/prepare_embedding_evaluation_suite.py`.

The public evaluator changes caption selection according to the requested
task set. Because the paper does not identify which branch produced Tables 2
and 3, the suite keeps four results separate:

| Protocol | Task | Queries | Source | Public-code basis |
|---|---|---:|---|---|
| `t2a_public_code_default_joint_all_captions` | T2A | 5,225 | `[CODE]` | default joint task set includes A2T, so all captions are used |
| `t2a_public_code_t2a_only_seed0` | T2A | 1,045 | `[CODE]` | T2A-only branch uses `random.Random(0).choice` once per clip |
| `t2t_public_code_default_seed0` | T2T | 1,045 | `[CODE]` | default `t2t_queries_per_clip=1` with effective seed 0 |
| `t2t_all_captions_sensitivity` | T2T | 5,225 | `[INFERRED]` | all-caption sensitivity analysis for the unpublished paper choice |

The seed-0 indices are materialized once with clip/query IDs and reused by
both one-caption protocols. No random choice occurs inside the canonical
evaluator. These variants must not be selected after observing Recall; all
four are reported with their protocol labels.

Each sub-run retains the canonical complete rankings and wrapper audit bundle.
Finalization independently rehashes every source, ranking, copied input,
config, log, Git record, and exit code, then writes `retrieval_summary.csv`
and `suite_metrics.json`. A changed or failed artifact stops finalization and
is recorded under `failures/`; completed suite metrics are never overwritten.
The four protocols require no GPU once embeddings exist and use less than
approximately 0.6 GB of additional result storage.

The AudioCaps-trained `OEA-Qwen3B` variant uses the identical four protocol
definitions through a separately bound suite:

```bash
bash scripts/run_qwen3b_ac_clotho_retrieval_suite.sh \
  /absolute/path/to/completed_qwen3b_ac_embedding_directory
```

Its source config is `configs/eval/qwen3b_clotho_retrieval_suite.json`. It
binds `results/model_locks/oea_qwen3b.json`, the `OEA-Qwen3B-AC` checkpoint,
and the full-generation protocol SHA256. The wrapper delegates only the
common CPU orchestration; the suite audit rejects a `Qwen3B-Cl` directory or
any mismatched model/checkpoint/protocol identity. This makes the Base versus
`+Cl` comparison use the same query selection and metric definitions without
conflating their model resources.

## Qwen3B-Cl Clotho positive-UIQ suite

The released Clotho Question, Imperative, Paraphrase, and `tagging` files each
contain exactly 1,045 unique target IDs. Their fixed byte sizes and SHA256
values are pinned in
`configs/eval/qwen3b_cl_clotho_positive_uiq_embeddings.json`. The paper calls
the fourth type Keyphrase while the release calls it `tagging`; both labels
are retained in every config, metadata row, summary, and result.

`scripts/run_qwen3b_cl_clotho_positive_uiq_embeddings.sh` generates 4,180
query embeddings in type-major and canonical-candidate order. Before loading
the model, it resolves the same committed caption protocol against the same
committed Qwen3B-Cl model lock and passes that stable resolved base config to
the UIQ generator. It reuses the
same `query:` prefix, chat template, last-hidden attention-mask mean pooling,
checkpoint text projection head, and L2 normalization as caption queries.
There is no UIQ-specific prompt. Before loading the model it requires every
released `original_captions` list to equal the corresponding canonical
Clotho manifest captions exactly. Source file drift, missing/extra target IDs,
schema changes, order changes, or caption changes are hard errors.

The UIQ generator is single-GPU, strict-offline, and resumable by the same
`RUN_ID`. Each chunk is written atomically and verified by shape, SHA256,
finite values, and unit norm. The committed batch size is 1 `[INFERRED]`; it
must be changed only through a new committed config after a measured smoke
test. Audio candidates are not re-encoded by this step.

After both the caption candidate bank and UIQ query bank are complete, the
CPU-only suite is launched with:

```bash
bash scripts/run_qwen3b_clotho_positive_uiq_suite.sh \
  /absolute/path/to/completed_caption_embedding_directory \
  /absolute/path/to/completed_uiq_embedding_directory
```

`scripts/prepare_positive_uiq_evaluation_suite.py` revalidates both generation
directories, the shared committed model lock, protocol/config identities, Git ancestry, every source and
artifact hash, metadata order, shapes, finite values, and unit norms. It then
materializes four explicit 1,045-row index lists and runs the canonical ID
evaluator separately for Tables 12–15. A query target must resolve to exactly
the fixed 1,045-candidate Clotho collection; no ID normalization, filtering,
or post-result protocol choice is allowed. Finalization rehashes all inputs,
rankings, logs, Git records, and exit codes before writing
`positive_uiq_summary.csv` and `suite_metrics.json`.

This completes the executable Clotho/Qwen3B-Cl path but does not yet constitute
a reproduced paper result: formal GPU embeddings have not run, the full four
CLAP plus six OEA model matrix is pending, and AudioCaps/MECAT candidates are
not ready. MECAT also remains blocked on the unpublished 847-versus-848
candidate choice.
