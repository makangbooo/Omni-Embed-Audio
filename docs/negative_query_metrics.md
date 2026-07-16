# Negative-query metric specification

This document fixes the metric contract used for Figure 3 and Table 17. It
does not resolve the missing target-to-hard-negative IDs in the released UIQ
files; it only makes the paper formulas executable and unit-tested.

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
Tests: `tests/test_negative_metrics.py`.
