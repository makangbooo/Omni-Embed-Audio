# OEA official-checkpoint reproduction report

Updated: 2026-08-11

## Scope

This report covers only the Omni-Embed-Audio paper reproduction. Results are
counted only when they are backed by a fixed checkpoint identity, a declared
evaluation protocol, completed artifacts, and a compact audit. Method
extensions that were developed after the reproduction are not part of this
tree or this report.

The main OEA results use the authors' released checkpoints. This project did
not reproduce the complete paper training run from raw training data.

## Claim policy

`PAPER` values, public-code protocols, inferred sensitivity protocols, and
missing details are kept separate. A public controlled result is not promoted
to a strict paper reproduction merely because its value is close to the
reported number.

The last whole-row inventory snapshot is `4/31 = 12.9%` strict completion.
The controlled model/dataset/task coverage is much larger and is recorded in:

- `results/tables/paper_experiment_matrix.csv`
- `results/tables/oea_partial_tables.md`
- `results/observations/reproduction_observations.jsonl`

## Model contract

The six evaluated OEA variants combine Nemotron-3B, Qwen2.5-Omni-3B, or
Qwen2.5-Omni-7B with the released AudioCaps or AudioCaps-plus-Clotho OEA
checkpoint. Runtime audits verify LoRA loading, separate text/audio projection
heads, masked mean pooling, L2 normalization, and 512-dimensional output
embeddings. Figure 2 evidence is fixed in
`results/audits/fig02_architecture_verification_20260728.json`.

Checkpoint revisions and resolved resource identities are stored under
`configs/checkpoints/`, `configs/eval/`, and `results/model_locks/`.

## Data boundary

| Dataset | Public evaluation input | Reproduction boundary |
|---|---:|---|
| Clotho v2.1 evaluation | 1,045 audio clips, 5 captions each | Public repaired release and UIQ IDs verified |
| AudioCaps test | 975 clips, 4,875 captions | Public train count is 91,254 versus paper 91,256 |
| MECAT public test | 848 clips | Paper reports 847; excluded ID is not public |
| UIQ positive | 11,472 queries | Four public query forms over three datasets |
| UIQ negative | 1,581 queries | Exact target/hard-negative audio pairing is not fully public |

No missing row, caption rule, or pairing is reconstructed from metric
proximity.

## Main retrieval results

The public-controlled Table 2/3 pipeline has been run for Clotho, AudioCaps,
and MECAT public-848 across the OEA variants, LAION-CLAP, M2D-CLAP,
MGA-CLAP, Robust-CLAP, and the three vanilla multimodal backbones.

Selected Clotho results are shown below. Values are percentages.

| Model and predeclared protocol | R@1 | R@5 | R@10 | Status |
|---|---:|---:|---:|---|
| OEA-Nemo3B-Cl T2A, all captions | 21.7225 | 47.1196 | 60.4402 | controlled/close |
| OEA-Nemo3B-Cl T2A, seed-0 caption | 21.6268 | 46.7943 | 59.8086 | controlled/close |
| OEA-Nemo3B-Cl T2T, seed-0 caption | 62.9665 | 75.9809 | 80.8612 | controlled/close |
| OEA-Qwen7B-AC T2A, all captions | 19.8852 | 44.6890 | 57.0909 | controlled |
| OEA-Qwen7B-Cl T2A, all captions | 22.0287 | 48.5359 | 61.6459 | controlled |
| vanilla Nemotron-3B T2A, all captions | 7.2536 | 21.3014 | 30.1818 | controlled/close |
| vanilla Qwen2.5-Omni-3B T2A, all captions | 0.1722 | 0.6316 | 1.1483 | controlled/close |
| vanilla Qwen2.5-Omni-7B T2A, all captions | 0.0957 | 0.6507 | 1.2440 | controlled |
| LAION-CLAP T2A, all captions | 14.0478 | 37.2057 | 49.8182 | controlled |
| M2D-CLAP T2A, all captions | 16.4211 | 40.7081 | 53.7033 | controlled |
| MGA-CLAP T2A, all captions | 21.1100 | 46.9474 | 60.0766 | controlled |

Strict Table 2/3 status remains blocked where the paper does not disclose
caption selection, T2T self-exclusion/tie handling, the MECAT 847-row
manifest, or the exact audio prefix path.

## Positive UIQ

The three-dataset, ten-model positive-UIQ controlled matrix is complete.
For OEA-Nemo3B-Cl on Clotho, the released query protocols give:

| Query form | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Question | 23.6364 | 50.8134 | 63.9234 |
| Imperative | 24.0191 | 51.1005 | 64.9761 |
| Paraphrase | 23.5407 | 49.9522 | 64.4019 |
| Keyphrase/tagging | 25.8373 | 52.6316 | 66.1244 |

The compact OEA evidence is in
`results/audits/nemo3b_cl_clotho_positive_uiq_eval_20260729.json` and the
corresponding model-specific UIQ audits.

## Negative UIQ and Tables 4/17

The controlled Negative UIQ matrix is complete for all ten models on Clotho,
AudioCaps, and MECAT public-848. The pairing audit covers all 1,581 released
negative queries. Pairing is reconstructed deterministically from caption
identity and explicit candidate aliases; embedding nearest-neighbor matching
is forbidden.

This establishes a reproducible public-data result, but not the unpublished
exact target/hard-negative pairing used by the authors. Evidence:

- `results/audits/negative_uiq_exact_pairing_audit_20260803.json`
- `results/audits/oea_negative_uiq_eval_20260803.json`
- `results/audits/clap_negative_uiq_eval_20260803.json`
- `results/audits/m2d_clap_negative_uiq_eval_20260803.json`

## Efficiency

Controlled efficiency measurements record warmup, batch, latency samples,
throughput, peak CUDA allocation/reservation, and parameter-count scope. A
representative RTX 4090 run for OEA-Nemo3B-Cl measured 530.315 ms per audio,
41.712 ms per text query, and 9.387 GiB peak allocated memory. M2D-CLAP and
MGA-CLAP were also measured under the same local hardware boundary.

The paper's A100 timing and memory procedure is incompletely specified, so
these results are controlled engineering comparisons rather than strict
replacements for Tables 5/16.

## Reproducibility failures encountered

1. Remote working-directory drift caused package import failures. Formal
   commands now enter the repository root before invoking Python.
2. `/home` and `/file_system` aliases differed across shells. Artifacts are
   identified by both absolute path and hash.
3. Checkpoint examples and actual filenames differed. Resource locks use the
   observed immutable revision and file identity.
4. PyTorch safe-loading changes and base tensors embedded in checkpoint
   states required structure-only inspection and verified extraction.
5. Clotho candidate filenames required explicit stem/suffix alias handling
   for Negative UIQ.
6. Partial terminal pastes were insufficient evidence. Completion requires
   status files, metrics, manifests, exit codes, and hashes.
7. Public data counts and paper counts differed for AudioCaps and MECAT. The
   public and paper protocols remain separate.

## Completion statement

The repository is sufficient to reproduce the public controlled OEA
official-checkpoint inference and evaluation workflow, including the main
retrieval tables, positive UIQ, Negative UIQ, baselines, and efficiency
audits, provided the external models, datasets, and recorded large artifacts
remain available. It is not evidence of a complete from-scratch OEA training
reproduction or of unpublished strict protocols.
