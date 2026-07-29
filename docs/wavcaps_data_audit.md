# WavCaps metadata and leakage audit (DATA-08/09)

Last updated: 2026-07-29 (Asia/Shanghai)

## Fixed source

- `[CODE]` The public source is `cvssp/WavCaps` at immutable revision `0930ec11ded28fa0eaa910fde2f6fc3538acbeac`.
- `[CODE][OBSERVED]` DATA-08 downloaded only the four JSON metadata files, three official blacklist files, and `README.md`: 8 files and 176,863,095 bytes in total. It did not download any audio archive.
- The complete public repository is approximately 819.51 GB. Audio remains a later resource and is not required for this metadata-only audit.
- The DATA-08 manifest pins every selected file by repository path, byte count, and SHA256/LFS object ID in `configs/resources/data08_wavcaps_metadata.json`.

## Source inventory

The remote DATA-08/09 wrapper `oea_takeover_data08_20260729_124908` completed at
commit `10466419e8450748247499eba090915709af8e74` with `FINAL_RUN_RC=0` in
130 seconds. The full DATA-09 pass parsed all 403,050 public metadata records:

| Source | Rows |
|---|---:|
| AudioSet_SL | 108,317 |
| BBC_Sound_Effects | 31,201 |
| FreeSound | 262,300 |
| SoundBible | 1,232 |
| **Total** | **403,050** |

## Duration-filter reconciliation

The paper reports 275,618 training samples and describes the duration filter as audio length `<=31` seconds. The pinned public metadata does not reproduce that count with the written predicate:

| Predicate over public metadata | Rows |
|---|---:|
| `duration < 31` | 275,624 |
| `duration <= 31` | 275,691 |
| `0 < duration < 31` | **275,618** |
| `0 < duration <= 31` | 275,685 |

There are 6 non-positive-duration records and 67 records with duration exactly 31 seconds. Therefore:

- `[PAPER]` reported count: 275,618.
- `[PAPER]` written condition: duration `<=31` seconds.
- `[INFERRED]` count-equivalent public-metadata condition: `0 < duration < 31` seconds.
- `[MISSING]` the exact filtered manifest used by the paper. DATA-09 does not silently replace the written paper predicate; it records all four counts.

## Leakage audit

### AudioCaps test against WavCaps AudioSet_SL

- `[CODE]` AudioCaps 2.0 test contains 975 unique YouTube IDs.
- Normalized exact YouTube-ID matching finds 173 WavCaps AudioSet_SL rows.
- All 173 rows satisfy the count-equivalent duration predicate.
- `[PAPER]` reports 173 overlaps, so this overlap count is exactly reproduced.

### Clotho evaluation against WavCaps FreeSound

- Clotho v2.1 evaluation contains 1,045 files.
- Case-insensitive filename matching finds 638 Clotho files, exactly reproducing the paper count.
- Those 638 filenames map to 1,017 WavCaps FreeSound rows because 64 matched filenames have more than one WavCaps candidate.
- 611 WavCaps rows are confirmed by both filename and FreeSound `sound_id`.
- A conservative all-candidate blocklist contains 1,017 rows, of which 383 satisfy the count-equivalent duration predicate.

The paper does not publish the exact Clotho-to-WavCaps blocklist or its duplicate-resolution rule. Consequently, 638 is an exact reproduced overlap count, while the 1,017-row conservative blocklist is `[INFERRED]` and must not be described as the paper's original blocklist.

### MECAT 00A/test against WavCaps AudioSet_SL

- Released MECAT positive UIQ contains 848 unique sample IDs representing 807 unique 11-character YouTube source-video IDs.
- Exact source-video ID matching against the pinned 108,317 WavCaps AudioSet_SL rows finds 4 shared videos and 4 MECAT samples.
- This is `[CODE][INFERRED]` provenance evidence. DATA-05 has now confirmed exact equality between the 848 archive IDs and released positive-UIQ IDs. A shared source video does not prove temporal or audio-content overlap because the WavCaps metadata does not publish the AudioSet_SL segment timestamp.
- `[MISSING]` the paper's audio/embedding model, threshold, candidate list, and manual review. DATA-11 reports the four candidates without applying a blocklist.

## Count after the metadata-only conservative blocklist

Applying the inferred `0 < duration < 31` filter and removing all 173 AudioCaps matches plus all 383 duration-eligible conservative Clotho candidates yields 275,062 rows. This is an `[INFERRED]` conservative reconstruction. The paper's exact post-blocklist training count is `[MISSING]`.

## Audit artifacts

The small summary is committed as `results/data_audits/data09_wavcaps_local_validation.json`. Full manifests remain outside Git:

| Artifact | Rows | Bytes | SHA256 |
|---|---:|---:|---|
| all public metadata | 403,050 | 262,733,469 | `ad1c1ce7e6294c985398ab8dd1a4e66ac3021f392e6fc39fefa6c48de4bb3ba7` |
| AudioCaps test blocklist | 173 | 35,984 | `77da39bcfa6283cff1fa59fdb0c012d16067e0dd6bfc3efdd2423ed4c1921a1d` |
| Clotho filename matches | 638 | 243,238 | `c40700763ce2094c909ee424cf3fa57b7057c2186bc2c0137f6313e7958c7cc4` |
| Clotho conservative candidates | 1,017 | 261,815 | `11c2a1d1f6e97f98f117dfb7c453419f554dd4bff4bdc724ee1c30ab88adb805` |
| filename + `sound_id` confirmations | 611 | 148,171 | `d6c51f269e8a81ef7b470d5f266595c6949573c6819928f930abc34fb6e7afb1` |

The full local validation was run twice; the second pass reused the outputs only after verifying deterministic hashes. Remote DATA-08/09 has now completed on the shared CPU storage and reported the same enforced count/protocol boundaries. Its wrapper evidence is fixed in `results/audits/data08_data09_wavcaps_remote_20260729.json`; child statistics and generated-artifact hashes remain pending read-only collection. DATA-11 next joins the canonical MECAT manifest to the pinned AudioSet_SL metadata and preserves the source-video candidate artifact separately from training blocklists.
