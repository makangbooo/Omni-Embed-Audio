# OEA data preparation

This document covers only datasets required by the OEA paper reproduction.
All generated audio, manifests, and embeddings live outside Git; the
repository stores scripts, fixed resource metadata, and compact audits.

## Clotho v2.1

### Evaluation split

The reproduction uses the repaired official Clotho v2.1 release from Zenodo
record 4783391. The paper names Clotho v2 but does not publish archive hashes,
so the Zenodo MD5 values are treated as the public immutable source identity.

Run in order:

```bash
bash scripts/download_data01_clotho_evaluation.sh
bash scripts/run_data02_clotho_validation.sh
```

The validator rechecks the archive, extracts through a staging directory,
requires exactly 1,045 WAV files and five captions per clip, fully decodes
every WAV, and verifies exact filename alignment for the four positive UIQ
sets. It audits the 542 Negative UIQ rows separately.

The canonical external manifest is:

`/home/jg525/datasets/oea/clotho_v2.1/manifests/clotho_evaluation_manifest.jsonl`

Each row records IDs, absolute/relative paths, captions, duration, sample
rate, channels, frames, decode status, and evaluation-only status.

### Training and validation archives

The optional Clotho train/validation inputs are prepared by:

```bash
bash scripts/download_data03_clotho_trainval.sh
bash scripts/run_data10_clotho_trainval_validation.sh
```

They are needed only for reproducing training or train/validation audits, not
for official-checkpoint evaluation. This project has not completed the paper's
from-scratch training reproduction.

## AudioCaps v2

Prepare and validate pinned metadata with:

```bash
bash scripts/download_data06_audiocaps_v2_metadata.sh
bash scripts/run_data07_audiocaps_v2_metadata_validation.sh
bash scripts/run_audiocaps_raw_audio_validation.sh
```

The evaluation protocol expects 975 test clips and 4,875 captions. Raw audio
is kept outside Git and must be decoded before embedding generation. The
public loader/manifest yields 91,254 training rows while the paper reports
91,256. Do not synthesize two rows or relabel the public manifest as the exact
paper training split.

Relevant compact evidence is under `results/audits/` with the
`audiocaps_*_20260802.json` names.

## MECAT

Prepare the public `00A/test` data with:

```bash
bash scripts/download_data04_mecat_00a_test.sh
bash scripts/run_data05_mecat_validation.sh
```

The public release contains 848 decodable FLAC files and aligns exactly with
the four released positive UIQ ID sets. The paper reports 847 pairs but does
not disclose the excluded sample or exact retrieval-caption construction.
Formal outputs must therefore be named `public848` and must not be presented
as the strict paper 847-row protocol.

See `docs/mecat_data_audit.md` and
`results/audits/data04_data05_mecat_remote_20260729.json`.

## WavCaps

WavCaps is required to audit the paper training-data description and potential
source overlap with evaluation data. Prepare metadata only with:

```bash
bash scripts/download_data08_wavcaps_metadata.sh
bash scripts/run_data09_wavcaps_metadata_audit.sh
bash scripts/run_data11_mecat_wavcaps_provenance.sh
```

The audit fixes source identities, duration-filter counts, and potential
same-video candidates. A common source video does not by itself prove audio
content overlap. The paper's exact training filter and post-blocklist manifest
remain separate from the public controlled inputs.

See `docs/wavcaps_data_audit.md` and the `data08_*`, `data11_*` audits.

## UIQ

The repository contains the released UIQ JSONL files under `data/UIQ/`:

| Dataset | Positive UIQ | Negative UIQ |
|---|---:|---:|
| AudioCaps | 3,900 | 630 |
| Clotho | 4,180 | 542 |
| MECAT public | 3,392 | 409 |
| Total | 11,472 | 1,581 |

Positive UIQ has four forms: question, imperative, paraphrase, and
keyphrase/tagging. Negative UIQ does not publish the complete exact
target/hard-negative audio pairing used in the paper.

The controlled reproduction reconstructs candidate aliases and caption
identity deterministically:

```bash
bash scripts/run_negative_uiq_pairing_audit.sh
```

Embedding similarity must never be used to guess the missing official pairing.
The controlled and strict pairing claims remain separate.

## Integrity rules

1. Pin source revision, filename, byte size, and available checksum before use.
2. Extract to a staging path and do not overwrite an incompatible final tree.
3. Decode every evaluation audio file and reject non-finite samples.
4. Require exact ID-set closure between audio, captions, queries, and qrels.
5. Record public/paper count differences rather than repairing them silently.
6. Keep raw data and embeddings outside Git; commit only manifests, hashes,
   configs, and compact audit evidence.
