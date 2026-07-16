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

This step does not extract the archive and does not constitute a dataset integrity pass. Extraction, WAV decoding, 1,045-row CSV checks, and UIQ `audio_id` alignment are performed in DATA-02.
