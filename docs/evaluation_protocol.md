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
