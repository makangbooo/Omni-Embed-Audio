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

## SQuTR

### DATA-12: immutable archive download

- `[CODE]` The official `SLLMCommunity/SQuTR` dataset is pinned to Hugging Face
  revision `2f1b041e2e98e0d28ed68fbcf22126ef247eb719`.
- The only archive is exactly 21,069,841,248 bytes with SHA256
  `8956bf938de3f9ce168a1e7daf2ff61b0b7fe603fa5c3d7dc6a4314617c6997c`.
- The remote download completed with both wrapper and downloader exit code 0.
  DATA-12 never extracts the archive.

### DATA-13A: read-only ZIP audit

The remote run `data13_squtr_archive_audit_20260726_135100` completed with exit
code 0. It found 149,349 ZIP records: 149,310 files, 39 directories, 42 JSONL
files, and 149,268 WAV files. The sum of uncompressed member sizes is
28,422,366,590 bytes. All six official subsets and all four acoustic
conditions match their expected query counts. No traversal, absolute path,
encrypted member, link, special member, duplicate path, or case-insensitive
collision was found.

`[OBSERVED]` All `queries_with_audio_*.jsonl` files are at the subset root and
all qrels are at `qrels/test.jsonl`. This resolves the public README/runner
path discrepancy from the actual pinned archive rather than by choosing the
more convenient code path.

### DATA-13B: resumable extraction and content audit

`scripts/run_data13b_squtr_validation.sh` is CPU-only and performs:

1. A second exact archive size/SHA256 check.
2. Safe streaming extraction with per-member CRC validation and atomic file
   promotion.
3. Exact reuse of valid existing files after an interruption; only
   tool-owned `.data13b.part` files may be restarted.
4. Refusal to overwrite any mismatched final file or traverse a symbolic-link
   directory.
5. Strict JSONL parsing, unique corpus/query IDs, qrels query/corpus closure,
   and full query coverage.
6. Exact four-condition audio filename/query-ID set equality, noisy-condition
   SNR/noise metadata checks, and first/last-frame WAV decoding after the full
   ZIP CRC pass.
7. An evaluation-only manifest with duration, sample rate, channels, path,
   original/normalized query text, condition, and audit fields.

Query-text differences are reported rather than silently rewritten: the
official benchmark documents a text-normalization stage, so exact equality
with the original text query is not assumed. The formal OEA-5 protocol remains
gated on a successful remote DATA-13B report.
