# Result management contract

This project keeps paper-reported numbers, reproduced observations, and the
generated comparison table as separate auditable artifacts. A missing remote
run is never converted into a reproduced value, and the builder never chooses
the most favorable evaluation protocol after seeing its metrics.

## Inputs and generated outputs

- `configs/results/paper_reported_metrics.jsonl` is the fixed paper registry.
  Each row has a stable `paper_metric_id`, a `[PAPER]` source tag, the PDF
  SHA256, the PDF page, the original two-decimal percentage, and the exact
  model/dataset/task/metric identity.
- `results/observations/reproduction_observations.jsonl` contains reviewed
  experimental observations. Numeric observations must name a repository-local
  JSON evidence file, its byte size and SHA256, and a `json_path` that resolves
  to exactly the declared reproduced value.
- `results/tables/reproduction_summary.csv` is generated and has the exact
  final-report columns requested by the reproduction protocol.
- `results/tables/reproduction_summary.audit.json` records both input
  identities, the output identity, row counts, unobserved paper metric IDs,
  status counts, and the delta formula.

Large embeddings, checkpoints, ranking arrays, datasets, and raw logs remain
outside ordinary Git. A reviewed observation may reference only a small JSON
result copied into the repository; the evidence identity prevents a later
silent edit from changing the comparison.

## Status policy

Only these final result statuses are accepted:

- `exact`: a numeric evidence-backed value that equals the paper value.
- `close`: a numeric evidence-backed value explicitly reviewed as close.
- `trend_reproduced`: a numeric evidence-backed value explicitly reviewed as
  reproducing the intended trend.
- `failed`: an attempted experiment with a fixed failure evidence file.
- `blocked`: no reproduced value because a required resource or protocol is
  unavailable.
- `not_reproducible`: no reproduced value after a documented determination
  that the experiment cannot be reproduced as stated.

The builder does not automatically assign `close` or `trend_reproduced` from a
numeric threshold. Those labels require an explicit review and note. A pending
experiment has no observation row; it is reported by the audit as an
unobserved paper metric rather than being mislabeled as `blocked`.

## Delta formulas

All paper and reproduced values use percentage units.

- `absolute_delta = abs(reproduced_value - paper_value)` in percentage points.
- `relative_delta = (reproduced_value - paper_value) / paper_value * 100` as a
  signed percent.

Generated deltas use decimal arithmetic and `ROUND_HALF_UP` to at most six
decimal places. Delta fields remain blank when no reproduced value exists.

## Build and validation

Run from the repository root in the `oea-repro` environment. This is CPU-only,
does not download anything, and writes only the two small generated result
files:

```bash
conda activate oea-repro
export CUDA_VISIBLE_DEVICES=""
python scripts/build_reproduction_summary.py
python -m unittest tests.test_build_reproduction_summary -v
```

The builder rejects duplicate observation identities, duplicate
experiment/metric/seed tuples, unknown paper metrics, non-finite values,
out-of-range percentages, evidence paths outside the repository, evidence
size/hash drift, unresolved JSON paths, and an `exact` label with unequal
values.

## Current incremental coverage

The first registry slice covers the already implemented OEA-Qwen3B (+Cl) on
Clotho paths:

- Table 2 T2A: R@1/R@5/R@10;
- Table 3 T2T: R@1/R@5/R@10;
- Tables 12–15 positive UIQ Question, Imperative, Paraphrase, and Keyphrase:
  R@1/R@5/R@10 for each query type.

This is 18 paper metrics. The six strict Table 2/3 metrics are currently
`blocked` because the paper and public evaluation command do not fully specify
caption selection and the T2T self/tie protocol. The four predeclared public
code protocols remain runnable and must be reported separately. The 12 UIQ
metrics are registered but unobserved until their official-checkpoint GPU
embedding and CPU retrieval runs finish. The released UIQ name `tagging` and
the paper name `Keyphrase` are both retained.

Additional paper rows are added to the registry only after their PDF values and
page locations are independently checked. Formal remote runs add observations
only after their small result JSON has been synchronized and hashed.
