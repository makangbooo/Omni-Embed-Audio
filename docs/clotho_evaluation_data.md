# Clotho evaluation data (DATA-01)

This is the first formal dataset resource used after the five-file repository smoke test.

## Fixed source

- `[INFERRED]` The paper says Clotho v2 but does not publish an archive checksum. We select the repaired official version 2.1 from Zenodo record 4783391 and report this version difference.
- `[INFERRED]` The selection is motivated by the official release notes stating that version 2.1 repairs corrupted files and illegal filename characters.
- `[PAPER][CODE]` The evaluation split contains 1,045 clips, each with five captions.
- `[MISSING]` The paper and official code do not publish archive checksums. The three MD5 values are therefore pinned separately from the official Zenodo file metadata in `configs/resources/data01_clotho_evaluation.json`.

The first step downloads only:

| File | Purpose | MD5 |
|---|---|---|
| `clotho_audio_evaluation.7z` | evaluation audio archive | `4569624ccadf96223f19cb59fe4f849f` |
| `clotho_captions_evaluation.csv` | five reference captions per clip | `1b16b9e57cf7bdb7f13a13802aeb57e2` |
| `clotho_metadata_evaluation.csv` | source metadata | `13946f054d4e1bf48079813aac61bf77` |

The download root is outside Git: `/home/jg525/datasets/oea/clotho_v2.1/source/` by default. Files ending in `.part` are incomplete resumable downloads. A verified `.part` is atomically renamed to its final filename. Existing final files are never overwritten: a checksum mismatch stops the run and preserves evidence.

DATA-01 does not extract the archive and does not itself constitute a dataset integrity pass. DATA-02 subsequently completed the extraction and full remote validation.

## DATA-02 remote validation result

Run `logs/data02_clotho_validation_20260717_233100` completed at Git commit `7a8fb5001c336929d5556fe2db7dda6f9212b25c` with a clean worktree and zero wrapper/validation exit codes. The validator reused a previously completed extraction only after verifying its completion marker.

- 1,045 caption rows, 1,045 metadata rows, 1,045 WAV files, and 1,045 successful decodes.
- Exactly five captions per audio file.
- 2,065,364,516 decoded-audio bytes and 23,416.309931972788 total seconds.
- Duration range 15.003174603174603–30.0 seconds; mean 22.407952088012237 seconds.
- Every file is mono at 44.1 kHz.
- Question, Imperative, Paraphrase, and `tagging` each contain exactly the same 1,045 IDs as the evaluation audio set.
- Negative UIQ contains 542 rows over 247 unique raw IDs. Appending `.wav` aligns all 542 rows, but this suffix rule is `[INFERRED]` and is not an official target/hard-negative pairing.
- Canonical manifest: 1,045 rows, 950,467 bytes, MD5 `253c1b275e3618fa94750150d7962da5`.

The committed audit summary is `results/data_audits/data02_clotho_evaluation_remote.json`. The full manifest remains outside Git at `/home/jg525/datasets/oea/clotho_v2.1/manifests/clotho_evaluation_manifest.jsonl`.
