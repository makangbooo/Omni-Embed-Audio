# Data preparation

## Clotho evaluation

### DATA-01: immutable source download

- `[INFERRED]` The paper names Clotho v2 but does not publish an archive checksum. The reproduction uses the repaired official Clotho v2.1 release from Zenodo record 4783391 and reports this version choice.
- `[PAPER][CODE]` The evaluation split contains 1,045 clips with five captions each.
- `[MISSING]` Archive checksums are absent from the paper and official code. DATA-01 pins the MD5 values published by the official Zenodo record.
- The three verified source files live outside Git at `/home/jg525/datasets/oea/clotho_v2.1/source/`.
- DATA-01 completed once with `downloaded_and_verified`; two repeat runs returned `verified_existing` and did not create duplicate copies.

### DATA-02: extraction and integrity validation

`scripts/run_data02_clotho_validation.sh` performs the following CPU-only steps:

1. Rechecks the 1.2 GB archive MD5 before extraction.
2. Detects `7zz`, `7z`, or `7za`; it does not install software automatically.
3. Extracts into a unique staging directory and refuses to overwrite an existing unmarked extraction.
4. Requires exactly 1,045 WAV files before atomically promoting the staging directory.
5. Requires exact 1,045-row captions and metadata CSV schemas and filename sets.
6. Fully decodes every WAV with SoundFile, checking frame count and finite samples.
7. Requires exact filename-set alignment for the 1,045 question, imperative, paraphrase, and tagging UIQ rows.
8. Audits the 542 negative UIQ rows separately.

The released negative JSONL has 542 rows but 247 unique raw `audio_id` values. The raw values omit `.wav`; appending the suffix aligns all 542 rows to positive Clotho filenames. This mapping is marked `[INFERRED]` and is not treated as an officially published target/hard-negative pairing.

The canonical generated manifest is stored outside Git at:

`/home/jg525/datasets/oea/clotho_v2.1/manifests/clotho_evaluation_manifest.jsonl`

Each row records the sample ID, absolute and relative audio path, five captions, split, duration, sample rate, channels, frames, format/subtype, existence/decode status, training inclusion status, filtering reason, and leakage-blocklist applicability. Evaluation samples are explicitly excluded from training; leakage blocklist status is `NOT_APPLICABLE_EVALUATION_SPLIT`, not an unsupported claim of no overlap.
