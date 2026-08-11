# Negative-query metric specification

This document fixes the metric and artifact contract used for Figure 3 and
Table 17. It does not resolve the missing target-to-hard-negative IDs in the
released UIQ files; it makes the paper formulas executable and provides a
strict evaluator for a pairing table obtained from an auditable source.

## Inputs and units

- `[PAPER]` `Rank(T)` is the one-based rank of the target audio in the complete
  candidate collection.
- `[PAPER]` `Rank(HN)` is the one-based rank of the paired hard-negative audio
  in the same collection and for the same query.
- `[CODE]` All rate metrics are returned on the 0-100 percentage scale used by
  the existing retrieval metrics and paper Table 17.
- `[MISSING]` The paper does not specify score-tie handling. The metric module
  therefore consumes already computed ranks and does not silently select a tie
  policy. A canonical evaluator must record its ranking/tie policy separately.

## Formulas

For `N` paired queries and a positive integer cutoff `k`:

| Output key | Definition | Source |
|---|---|---|
| `R@k` | `mean(Rank(T) <= k) * 100` | `[PAPER]` standard retrieval |
| `Delta-Rank` | `mean(Rank(HN) - Rank(T))` | `[PAPER]` §3.3, Appendix L |
| `HNSR` | `mean(Rank(HN) > Rank(T)) * 100` | `[PAPER]` Appendix L |
| `HNSR@k` | `mean((Rank(T) <= k) AND (Rank(HN) > k)) * 100` | `[PAPER]` §3.3, Figure 3 |
| `TFR` | `mean(Rank(T) == 1) * 100` | `[PAPER]` Appendix L |
| `TFR-HN@k` | `mean((Rank(T) == 1) AND (Rank(HN) > k)) * 100` | `[PAPER]` Figure 3, Appendix L |

Higher is better for all listed metrics, including `Delta-Rank`.

## Paper naming inconsistency

`[PAPER]` Figure 3 and Appendix L call the strict combined metric
`TFR-HN@k`. Table 17 labels its final column `TFR@10`, while the table caption
calls the strict metrics `TFR/TFR@10`. The implementation retains the explicit
key `TFR-HN@k`; when reproducing Table 17, `TFR@10` must be treated only as a
display alias for `TFR-HN@10`, not as a separate formula.

## Guardrails verified by synthetic tests

- Moving the target toward rank 1 improves the relevant metrics.
- Moving the hard negative outside top-k improves suppression metrics.
- A target outside top-k cannot count as `HNSR@k`, even if it is ranked above
  the hard negative.
- `TFR-HN@k` requires both target rank 1 and hard-negative rank greater than k.
- Equal target and hard-negative ranks do not count as suppression.
- Empty, non-integral, non-finite, zero-based, mismatched, or duplicate-cutoff
  inputs fail explicitly.

Implementation: `AudioRetrieval/evaluation/negative_metrics.py`.

## Explicit-pair embedding evaluator

`AudioRetrieval/evaluation/negative_canonical.py` computes target and
hard-negative ranks from a fixed query embedding bank and a fixed candidate
embedding bank. `scripts/evaluate_negative_embedding_artifacts.py` validates
the inputs and writes the audit bundle. The CPU wrapper is
`scripts/run_negative_embedding_evaluation.sh`.

The evaluator enforces these rules:

1. Query and candidate IDs are non-empty and unique.
2. A separate JSONL pairing table must contain exactly one row for every
   query, with explicit `query_id`, `target_id`, and `hard_negative_id`.
3. Pairing query IDs must exactly equal the query-metadata ID set. Missing,
   extra, or duplicate rows fail; a selected query subset does not weaken this
   full-bank coverage check.
4. Both audio IDs must exist in the candidate metadata and must differ.
5. No caption, filename, score, nearest neighbor, or released
   `negative_captions` field is used to infer an audio ID.
6. Embeddings must be finite 2-D matrices with matching dimensions. When
   normalization is enabled, zero-norm rows fail.
7. Both metric ranks use `optimistic_strict_greater`, matching the public
   retrieval code. The separately saved full ranking uses descending score
   with candidate index as the deterministic tie breaker.
8. Existing artifacts are never overwritten. Formal runs require a clean Git
   worktree by default and record that commit/status.

The minimum configuration is:

```json
{
  "schema_version": 1,
  "experiment_id": "oea_qwen3b_clotho_negative_seed42_YYYYMMDD_HHMMSS",
  "model": "OEA-Qwen3B (+Cl)",
  "checkpoint": "/absolute/path/to/step_40.pt",
  "dataset": "Clotho negative UIQ",
  "task": "negative",
  "paper_table": "Table 17",
  "protocol_label": "author-pairing-v1",
  "protocol_source": "CODE",
  "seed": 42,
  "ks": [1, 5, 10],
  "query_embeddings": "/absolute/path/to/query_embeddings.npy",
  "candidate_embeddings": "/absolute/path/to/candidate_embeddings.npy",
  "query_metadata": "/absolute/path/to/query_metadata.jsonl",
  "candidate_metadata": "/absolute/path/to/candidate_metadata.jsonl",
  "pairing_metadata": "/absolute/path/to/pairing_metadata.jsonl",
  "query_selection": "all"
}
```

`protocol_source` describes the source of the complete protocol, not merely
the metric formulas. It must not be set to `PAPER` unless the exact pairing
and candidate set are supported by paper/author evidence.

A complete run saves copied embeddings and three metadata JSONLs; input and
artifact SHA256 values; full similarities and rankings; target/HN indices and
one-based ranks; selected query indices; per-query scores, ranks, conditions,
and delta; aggregate metrics; resolved config; command; environment; Git
state; stdout/stderr; and exit code. Failed runs retain `metrics.json` with the
exception and traceback.

The evaluator itself enforces explicit-ID alignment, exact pairing coverage,
duplicate rejection, target/HN inequality, deterministic ties, and
non-overwrite behavior. Completed run evidence is retained under
`results/audits/` and `results/raw/`.
