# Result management contract

This project keeps paper-reported numbers, reproduced observations, and the
generated comparison table as separate auditable artifacts. A missing remote
run is never converted into a reproduced value, and the builder never chooses
the most favorable evaluation protocol after seeing its metrics.

## Inputs and generated outputs

- `configs/results/paper_reported_tables.json` is the human-reviewable source
  matrix. It classifies every paper table from Table 1 through Table 17 and
  records every displayed value as a string so that the original precision is
  preserved.
- `configs/results/paper_reported_metrics.jsonl` is generated from that matrix.
  It contains 910 stable paper metric identities from all 13 numeric result
  tables, with `[PAPER]` provenance, PDF SHA256/page, unit, displayed value,
  sort order, and complete experiment identity.
- Tables 6, 8, 9, and 10 are explicitly classified as non-metric tables:
  categorical provenance, UIQ generation configuration, qualitative examples,
  and LLM evaluation configuration, respectively. They are not silently
  omitted from the table inventory.
- `results/paper_audits/paper_metric_transcription.json` records the independent
  PDF transcription audit. All 910 configured numeric cells must match the row
  label and complete displayed number sequence on the pinned PDF page.
- `results/observations/reproduction_observations.jsonl` contains reviewed
  experimental observations. Numeric observations must name a repository-local
  JSON evidence file, its byte size and SHA256, and a `json_path` that resolves
  to exactly the declared reproduced value.
- `results/tables/reproduction_summary.csv` is generated and has the exact
  final-report columns requested by the reproduction protocol.
- `results/tables/reproduction_summary.audit.json` records input and output
  identities, row counts, units, unobserved paper metric IDs, status counts,
  and the delta formula.

Large embeddings, checkpoints, ranking arrays, datasets, and raw logs remain
outside ordinary Git. A reviewed observation may reference only a small JSON
result copied into the repository; the evidence identity prevents a later
silent edit from changing the comparison.

## Status policy

Only these final result statuses are accepted:

- `exact`: an evidence-backed value equal to an exact paper value.
- `close`: an evidence-backed value explicitly reviewed as close.
- `trend_reproduced`: an evidence-backed value explicitly reviewed as
  reproducing the intended trend.
- `failed`: an attempted experiment with a fixed failure evidence file.
- `blocked`: no reproduced value because a required resource or protocol is
  unavailable.
- `not_reproducible`: no reproduced value after a documented determination
  that the experiment cannot be reproduced as stated.

The builder does not automatically assign `close` or `trend_reproduced` from a
numeric threshold. Those labels require an explicit review and note. A pending
experiment has no observation row; it is reported by the audit as an
unobserved paper metric rather than being mislabeled as `blocked`. A paper cell
marked with a qualifier such as `approximately` cannot receive `exact`, even if
the displayed number is numerically equal.

## Units and delta formulas

The registry retains the unit declared by each table: percent,
`score_1_to_5`, score standard deviation, milliseconds, gigabytes, million
parameters, count, or rank gap. Mean scores and their standard deviations use
distinct units and validation ranges. Displayed paper precision is preserved
instead of normalizing every value to two decimals.

- `absolute_delta = abs(reproduced_value - paper_value)` in the registry unit.
- `relative_delta = (reproduced_value - paper_value) / paper_value * 100` as a
  signed percent.

Generated deltas use decimal arithmetic and `ROUND_HALF_UP` to at most six
decimal places. Delta fields remain blank when no reproduced value exists.

## Build and validation

Run from the repository root in the `oea-repro` environment. These commands
are CPU-only and do not download anything. The first two maintain the generated
registry; the summary builder writes only the two small result-table artifacts.

```bash
conda activate oea-repro
export CUDA_VISIBLE_DEVICES=""
python scripts/build_paper_metric_registry.py --check
python scripts/build_reproduction_summary.py
python -m unittest tests.test_build_paper_metric_registry \
  tests.test_build_reproduction_summary -v
```

When the pinned paper PDF is locally available, rerun the independent
transcription audit before changing any paper value:

```bash
python scripts/audit_paper_metric_transcription.py \
  --pdf /path/to/Omni-Embed-Audio.pdf
```

The audit verifies the PDF SHA256, rejects a stale generated registry, checks
every configured numeric row against its pinned PDF page, and stores no
extracted copyrighted table text. The builder rejects duplicate identities,
dimension/count drift, unknown units, invalid unit ranges, non-finite values,
evidence paths outside the repository, evidence size/hash drift, unresolved
JSON paths, and invalid `exact` labels.

## Current coverage

All 17 paper tables are classified. The 13 numeric result tables contribute
910 registered paper metrics, and the PDF transcription audit currently
matches 910/910 cells. The Qwen3B-Cl/Clotho T2A/T2T run contributes 12
reviewed `close` observations from four predeclared public-code/sensitivity
protocols, and its positive-UIQ run contributes another 12 reviewed `close`
observations for Tables 12–15. The AudioCaps-trained Qwen3B/Clotho run adds 12
reviewed `close` T2A/T2T observations under the same four protocols and 12
reviewed `close` positive-UIQ observations for Tables 12–15.
Twelve strict Table 2/3 observations across the two variants remain `blocked`
because the paper and public evaluation command do not fully specify caption
selection and the T2T self/tie protocol. The remaining 874 registered metrics are unobserved
pending formal runs; they are not represented as reproduced values.

The vanilla Nemotron-3B/Clotho main-table run adds 12 reviewed `close`
observations under the same four-protocol separation, bringing the committed
snapshot to 155 observations: 132 `close` and 23 strict `blocked`.

The four predeclared public-code retrieval protocols remain runnable and must
be reported separately. The released UIQ name `tagging` and the paper name
`Keyphrase` are both retained. Formal remote runs add observations only after
their small JSON evidence has been synchronized, reviewed, and hashed.
