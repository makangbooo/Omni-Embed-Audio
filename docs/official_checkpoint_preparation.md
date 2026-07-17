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
7. reload with `weights_only=True`, select only keys containing `lora_`, and
   write a new `_inference_only.pt` file without overwriting an existing file;
8. reload the derived file and compare every LoRA and projection tensor for
   exact equality;
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

# Structure and identity audit only:
bash scripts/run_official_checkpoint_preparation.sh oea_nemo3b --inspect-only

# Audit followed by non-overwriting inference-only extraction:
bash scripts/run_official_checkpoint_preparation.sh oea_nemo3b
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
  deletion is used.

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

Formal model-specific embedding configs may be committed only after this
report supplies the derived checkpoint SHA256 and confirms the actual LoRA and
projection structure.
