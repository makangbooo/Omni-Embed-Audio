# Official OEA checkpoint preparation

## Purpose

The six author-released OEA checkpoints are large training checkpoints that
also contain frozen backbone tensors. Formal evaluation uses a smaller derived
artifact containing only the LoRA tensors, audio projection head, text
projection head, and non-tensor training metadata. The derived file is a local
reproduction artifact; it is never described as an author checkpoint.

The public README says that every checkpoint repository contains
`step_40.pt`. The immutable repositories contradict that statement for five of
the six variants. Reproduction therefore uses the pinned revision and LFS
identity, not a guessed step number.

## Fixed source identities

All identities below are `[CODE]`, read from immutable official Hugging Face
revision metadata and duplicated in the corresponding resource manifests.

| Variant | Official source file | Bytes | LFS SHA256 prefix |
|---|---|---:|---|
| OEA-Nemo3B | `OEA-Nemo3B-AC/step_400_best.pt` | 9,466,826,153 | `55579dfbd4f6` |
| OEA-Nemo3B (+Cl) | `OEA-Nemo3B-Cl/step_450_best.pt` | 9,466,826,217 | `c90132858941` |
| OEA-Qwen3B | `OEA-Qwen3B-AC/step_350.pt` | 9,466,835,918 | `afb22d02e610` |
| OEA-Qwen3B (+Cl) | `OEA-Qwen3B-Cl/step_40.pt` | 9,466,833,858 | `d5f2648c19b0` |
| OEA-Qwen7B | `OEA-Qwen7B-AC/step_300.pt` | 17,940,602,533 | `cd751e3a71f0` |
| OEA-Qwen7B (+Cl) | `OEA-Qwen7B-Cl/step_330.pt` | 17,940,602,661 | `09ce41af7e7a` |

The complete revisions and 64-character hashes live in
`configs/checkpoints/official_oea_checkpoints.json` and
`configs/resources/model01_qwen3b_cl.json` through
`configs/resources/model04_qwen7b.json`.

## Audited workflow

For one variant, the runner performs these steps in order:

1. resolve the model and checkpoint only through the committed six-variant
   registry;
2. reject a model root inside the Git repository and reject unsafe paths;
3. verify the fixed source byte size;
4. hash the entire source and compare its SHA256 to the official LFS identity;
5. load with `weights_only=True`, `mmap=True`, and `FakeTensorMode` to audit
   top-level sections, unsafe globals, tensor keys, shapes, dtypes, and sizes;
6. take the LoRA tensor count and byte count from that exact inspection rather
   than inferring them from an architecture name;
7. either reload with `weights_only=True`, select only keys containing `lora_`,
   and write a new `_inference_only.pt` file without overwriting an existing
   file, or explicitly select read-only verification of an existing derived
   artifact;
8. load the derived file and compare its exact top-level schema, source SHA256,
   metadata, every LoRA tensor, and every projection tensor with the fixed raw
   checkpoint;
9. record source/destination hashes, commands, Git state, environment logs,
   memory, exit codes, and failures.

An inspection mismatch stops before extraction. A failed extraction removes
only its uniquely named temporary file; it does not alter the official source,
other checkpoints, datasets, or prior derived artifacts.

## Command

Run on a CPU server after the selected source checkpoint has passed the model
resource audit:

```bash
cd /home/jg525/Omni-Embed-Audio
conda activate oea-repro
export MODEL_ROOT=/home/jg525/model_cache/oea

# Read-only base/checkpoint snapshot audit for this exact variant:
bash scripts/run_official_oea_model_resource_audit.sh oea_nemo3b

# Structure and identity audit only:
bash scripts/run_official_checkpoint_preparation.sh oea_nemo3b --inspect-only

# Audit followed by non-overwriting inference-only extraction:
bash scripts/run_official_checkpoint_preparation.sh oea_nemo3b

# For an artifact previously produced by the audited extraction script:
bash scripts/run_official_checkpoint_preparation.sh \
  oea_qwen3b_cl --verify-existing-derived
```

Valid IDs are:

```text
oea_nemo3b
oea_nemo3b_cl
oea_qwen3b
oea_qwen3b_cl
oea_qwen7b
oea_qwen7b_cl
```

## Resources and outputs

- Server: CPU; `CUDA_VISIBLE_DEVICES` is empty.
- Environment: `oea-repro`.
- Network: none after the checkpoint is present.
- Source reads: approximately twice the raw checkpoint size because inspection
  and extraction independently verify it.
- Memory: mmap/FakeTensor inspection is designed to avoid materializing the
  frozen backbone; actual peak RSS is recorded and not assumed in advance.
- Time: `[INFERRED]` 10–45 minutes for a 3B checkpoint and 20–90 minutes for a
  7B checkpoint, depending mainly on shared-storage throughput.
- Disk: the source is not copied; only a small derived LoRA/projection artifact
  and logs are added. The exact size is recorded rather than predicted.
- Overwrite/deletion risk: existing derived outputs are rejected; no recursive
  deletion is used. `--verify-existing-derived` requires the destination to
  exist and never writes it.

Each run creates a timestamped ignored directory:

```text
logs/official_checkpoint_preparation_<variant>_<timestamp>/
├── preparation_manifest.json
├── checkpoint_inspection.json
├── extraction_manifest.json        # full mode only
├── command.sh
├── git_commit.txt
├── git_status.txt
├── preparation_exit_code.txt
├── wrapper_exit_code.txt
├── stdout.log
├── stderr.log
└── memory/disk snapshots
```

## Portable model lock

Formal evaluation must not copy model identities manually from download logs.
After both the selected base/checkpoint assets and the derived checkpoint have
passed the workflows above, build a small portable lock:

```bash
python scripts/build_official_oea_model_lock.py \
  --variant oea_qwen3b_cl \
  --model-resource-audit logs/model_resource_audit_YYYYMMDD_HHMMSS/model_resource_audit.json \
  --checkpoint-preparation logs/official_checkpoint_preparation_oea_qwen3b_cl_YYYYMMDD_HHMMSS/preparation_manifest.json \
  --output results/model_locks/oea_qwen3b_cl.json
```

The builder accepts an overall `incomplete` resource audit only when the two
assets selected by the requested variant are individually complete. This
allows one multi-asset audit to contain an unrelated unfinished download
without weakening the selected evidence chain. For both selected assets it
requires the exact immutable repository revision, revision marker, destination,
file count, byte count, and per-file Git-blob or LFS verification. It also
rehashes the derived checkpoint and rejects any drift after extraction.

The resulting JSON contains no server-absolute model path. It locks every base
snapshot file, the official raw checkpoint identity, the derived checkpoint
identity, measured LoRA structure, both input-report hashes, and both evidence
Git commits. Generation requires a clean worktree and refuses to overwrite an
existing output.

This step uses the `oea-repro` environment on a CPU server, disables no files,
downloads nothing, and does not require a GPU. It reads only the small derived
checkpoint because the large base and raw checkpoint were already hashed by
the two cited audit reports. Expected time is under one minute and the output
is a small JSON file. Formal model-specific embedding configs may be committed
only after this lock supplies the derived SHA256 and confirms the measured LoRA
and projection structure.

## One-command per-variant CPU pipeline

After a variant's raw resources are present, the three formal gates can run in
one failure-stopping command:

```bash
# New derived artifact:
bash scripts/run_official_oea_model_pipeline.sh oea_nemo3b

# Existing Qwen3B-Cl artifact from the earlier audited extraction:
bash scripts/run_official_oea_model_pipeline.sh \
  oea_qwen3b_cl --verify-existing-derived
```

The stages are fixed as: exact variant-scoped resource audit, raw checkpoint
inspection plus extract/verify, then model-lock generation. A nonzero stage
prevents every later stage and remains in `pipeline_manifest.json`. All large
intermediate reports stay in the ignored timestamped `logs/` directory. The
only repository-visible output is
`results/model_locks/<variant_id>.json`; an existing lock is never overwritten.

This is a CPU task with `CUDA_VISIBLE_DEVICES` empty and no model download. It
does use Hugging Face metadata access and reads every selected base-model file
once plus the raw checkpoint multiple times. Estimated total reads are roughly
40 GB for a 3B variant and 76 GB for a 7B variant `[CODE][INFERRED]`; expected
time is 30–150 minutes `[INFERRED]`, dominated by shared-storage throughput.
Therefore each real execution must receive the required long-operation resource
report before it starts. A successful run intentionally creates a small
uncommitted lock for review; the model, raw checkpoint, derived weights, and
large logs remain outside Git.

The reviewed lock must then be committed and pushed as its own small evidence
artifact before formal GPU work. Caption and positive-UIQ wrappers resolve the
committed protocol against that committed lock with
`scripts/build_official_oea_eval_config.py`. The resolver expands the runtime
configuration to the complete locked base-file inventory, records both input
hashes and the current Git commit, creates or exactly verifies a stable result,
and rejects an untracked lock. The GPU generator independently reopens both
tracked inputs and rejects any identity, variant, resource, or commit drift.
