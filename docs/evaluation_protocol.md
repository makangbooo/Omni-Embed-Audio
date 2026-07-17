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

`scripts/generate_oea_embeddings.py` and
`scripts/run_qwen3b_cl_clotho_embeddings.sh` implement the first formal
OEA-Qwen3B (+Cl) embedding pass over the canonical Clotho evaluation manifest.
The fixed configuration is
`configs/eval/qwen3b_cl_clotho_embeddings.json`.

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
