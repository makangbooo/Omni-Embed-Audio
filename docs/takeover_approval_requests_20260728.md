# OEA takeover approval requests (2026-07-28)

This file covers only original OEA paper experiments. Nothing listed here has
been downloaded or executed in this takeover.

## Remote access prerequisite

The GitHub mirror is clean and synchronized at
`repro/oea-full@4a7852e0d35e544ea87fb1d5a7eef4f44493714d`. Read-only audit of
`/home/jg525/Omni-Embed-Audio`, `/home/jg525/models/oea`,
`/home/jg525/datasets/oea`, `/home/jg525/experiment_cache`, logs, raw results,
and tmux is blocked by SSH authentication. Alias `bitahub_JG` also reports a
changed ED25519 host key:

- newly presented: `SHA256:etC2qN4P9phlmmEtoHi7hO3qfnmevz/u7bJv8Q1JwlM`
- stored for port 42017: `SHA256:+lMykBVk8nCgA/Aav7/pG5AcS6UrJqy5OuKKLt+q8ZA`

No `known_hosts` entry will be changed and host-key verification will not be
bypassed without explicit confirmation of the new fingerprint. A working SSH
authentication method is also required; no usable local agent/key is present.

## Download request DATA-04: MECAT 00A test

- Item: `mispeech/MECAT-Caption`, only `README.md` and
  `00A/test_0000-0000000.tar.gz`.
- Exact URL: `https://huggingface.co/datasets/mispeech/MECAT-Caption/resolve/be4a24c3f7309d74208e08a7cce49e72cb7a5834/00A/test_0000-0000000.tar.gz`.
- Revision: `be4a24c3f7309d74208e08a7cce49e72cb7a5834`.
- License: CC BY 3.0, from the pinned Hugging Face data card.
- Download size: 173,183,713 bytes total (173,168,424-byte shard plus
  15,289-byte README).
- Extracted estimate: approximately 174-220 MB `[INFERRED]`; DATA-04 itself
  downloads only, and DATA-05 measures exact member bytes before extraction.
- Absolute destination: `/home/jg525/datasets/oea/mecat_caption_be4a24c3/source`.
- Need: public 848-item archive validation, decode/UIQ-ID audit, and provenance;
  it does not resolve PAPER's unpublished 847-item subset.
- Smaller alternative: the released UIQ metadata is already present, but it
  cannot validate audio or the canonical public archive. This one shard is the
  smallest audio-bearing official scope.
- Resume: yes; Hugging Face partial cache is preserved and every final file is
  checked against size/LFS SHA256.
- Exact command after approval and remote read-only precheck:
  `bash scripts/download_data04_mecat_00a_test.sh`.

## Download request DATA-06: AudioCaps 2.0 metadata

- Item: official `dataset2.0/{README.md,train.csv,val.csv,test.csv}`.
- Exact source: `https://github.com/cdjkim/audiocaps/tree/d004db3ea1b01cf4fd0347dd8d27db90cadc8809/dataset2.0`; four raw URLs and
  hashes are fixed in `configs/resources/data06_audiocaps_v2_metadata.json`.
- Revision: `d004db3ea1b01cf4fd0347dd8d27db90cadc8809`.
- License: MIT for the official repository; underlying YouTube audio is not
  included and remains subject to source rights/official access terms.
- Download and installed size: 6,879,660 bytes.
- Absolute destination: `/home/jg525/datasets/oea/audiocaps_v2_d004db3/metadata`.
- Need: formal DATA-06/07 schema/count/hash audit and the 91,256 versus 91,254
  conflict record. This does not claim AudioCaps audio availability.
- Smaller alternative: `test.csv` alone is 397,498 bytes and suffices for some
  evaluation joins, but not the paper's training-data/count audit.
- Resume: yes; `.part` files are preserved and final MD5/SHA256/bytes are
  verified without overwriting mismatches.
- Exact command after approval and remote read-only precheck:
  `bash scripts/download_data06_audiocaps_v2_metadata.sh`.

## Download request DATA-08: WavCaps metadata only

- Item: eight metadata/official-blacklist files; every `Zip_files/**` audio
  archive is excluded.
- Exact repository: `https://huggingface.co/datasets/cvssp/WavCaps`.
- Revision: `0930ec11ded28fa0eaa910fde2f6fc3538acbeac`.
- License: CC BY 4.0, from the pinned Hugging Face data card.
- Download/installed size: 176,863,095 bytes.
- Absolute destination: `/home/jg525/datasets/oea/wavcaps_0930ec11/source`.
- Need: canonical DATA-09 overlap/provenance and duration/filter-count audit.
- Smaller alternative: AudioSet_SL + FreeSound + relevant blacklist files can
  check the two known overlaps, but all eight fixed files are required for the
  four-subset count and paper training-pool audit.
- Resume: yes; Hugging Face partial cache is preserved and all pinned files are
  verified by exact bytes/SHA256.
- Exact command after approval and remote read-only precheck:
  `bash scripts/download_data08_wavcaps_metadata.sh`.

## Deferred model candidate: OEA-Qwen7B-Cl

This is not yet submitted for download approval because the remote model tree
has not been re-audited and a duplicate 17.94 GB transfer would be unjustified.
If the read-only audit confirms it is absent/incomplete, the candidate is:

- Repository: `https://huggingface.co/JudeJiwoo/OEA-Qwen7B-Cl`.
- Revision/file: `30c6e97cfdf451b1948013d2839befe0c3022c46/step_330.pt`.
- License: MIT; size 17,940,602,661 bytes; LFS SHA256
  `09ce41af7e7ac23106fa74d25e45b7229a046d687774cbd4ab2c00c05097d92e`.
- Destination: `/home/jg525/models/oea/OEA-Qwen7B-Cl/step_330.pt`.
- No smaller inference-only artifact is officially published; the repository
  pipeline derives a non-overwriting inference-only artifact after audit.
- Hugging Face partial download is resumable.

The four CLAP baselines cannot yet form valid download requests: the paper and
public repository do not pin exact LAION/Robust/MGA/M2D checkpoint identities,
revisions and SHA256 values. No approximate checkpoint will be substituted.

## GPU request: Nemo3B-Cl Clotho positive UIQ

- Paper experiment/stage: `EXP-12` through `EXP-15`, released Question,
  Imperative, Paraphrase and Keyphrase/tagging UIQ on Clotho.
- Model/checkpoint: `JudeJiwoo/OEA-Nemo3B-Cl@9588912298afca0b11f5895b864ae28083f35022`, audited
  `step_450_best_inference_only.pt`, SHA256
  `2a5bee9039a28c0028cf205d1e2f4302fda540dd913f4cc1b08301edfc6680c4`.
- Dataset: fixed Clotho v2.1 evaluation manifest, 1,045 candidates and 4,180
  released positive UIQ queries.
- GPU: one RTX 4090 24 GB; expected approximately 9.2 GiB allocated and no more
  than 10.5 GiB reserved, with 12 GiB requested headroom.
- Runtime: GPU generation 6-15 minutes; CPU finalization 2-10 minutes; total
  8-25 minutes. Basis: two Qwen3B 4,180-query runs completed in 312-334 seconds
  on RTX 4090, plus Nemo/model-load and CPU margin.
- Cache/result directories:
  `/home/jg525/Omni-Embed-Audio/results/raw/oea_nemo3b_clotho_positive_uiq_embeddings_seed42_20260728_takeover01`
  and
  `/home/jg525/Omni-Embed-Audio/results/raw/oea_nemo3b_clotho_positive_uiq_suite_seed42_20260728_takeover01`.
- Log directory:
  `/home/jg525/Omni-Embed-Audio/logs/oea_nemo3b_clotho_positive_uiq_20260728_takeover01`.
- Resume: yes. Each query chunk is written atomically and hash-verified; reuse
  requires the identical run identity. Interruption preserves complete chunks
  and cannot start the CPU suite until generation exits zero.
- Failure handling: retain failure JSON, attempt logs and completed chunks;
  inspect without changing the protocol; retry the same RUN_ID only when the
  identity matches, otherwise use a new run directory. Existing results are
  never overwritten.
- Exact experiment command inside the tmux pane:

```bash
RUN_ID=oea_nemo3b_clotho_positive_uiq_embeddings_seed42_20260728_takeover01 \
SUITE_ID=oea_nemo3b_clotho_positive_uiq_suite_seed42_20260728_takeover01 \
TMUX_LOG_DIR=/home/jg525/Omni-Embed-Audio/logs/oea_nemo3b_clotho_positive_uiq_20260728_takeover01 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_nemo3b_cl_clotho_positive_uiq_tmux.sh \
  /home/jg525/Omni-Embed-Audio/results/raw/oea_nemo3b_clotho_embeddings_seed42_20260727_094554
```

The launcher refuses non-tmux execution, tees stdout/stderr to the visible pane
and log, prints progress/elapsed/throughput/ETA, emits the required final status
block, and then keeps the pane in interactive Bash.

## Ready CPU task after SSH access

The missing Nemo3B-Cl Clotho T2T metrics require no download or GPU. The
two-protocol finalizer intentionally excludes the already completed T2A runs.
Expected CPU runtime is 5-15 minutes and the proposed unique command is:

```bash
SUITE_ID=oea_nemo3b_clotho_t2t_suite_seed42_20260728_takeover01 \
TMUX_LOG_DIR=/home/jg525/Omni-Embed-Audio/logs/oea_nemo3b_clotho_t2t_20260728_takeover01 \
bash scripts/run_nemo3b_cl_clotho_t2t_tmux.sh \
  /home/jg525/Omni-Embed-Audio/results/raw/oea_nemo3b_clotho_embeddings_seed42_20260727_094554
```

The launcher refuses non-tmux execution, prints per-protocol elapsed,
throughput and ETA fields, tees both streams to the visible pane and combined
log, emits a final status block, and retains interactive Bash after completion.
