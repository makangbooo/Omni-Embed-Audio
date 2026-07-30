# Vanilla multimodal-backbone evaluation

The paper reports three untrained multimodal-LLM backbones as retrieval baselines: Nemotron-3B, Qwen2.5-Omni-3B, and Qwen2.5-Omni-7B. A vanilla result must not load an OEA checkpoint, LoRA weights, or either 512-dimensional projection head.

## Fixed public-code protocol

| Component | Fixed value | Evidence |
| --- | --- | --- |
| Text input | public runtime renders `query:<caption>` with no inserted separator | `[PAPER]` prefix; `[CODE]` exact concatenation |
| Audio input | audio-only chat message, with no inserted text prefix | `[CODE]` |
| Paper/audio conflict | paper states `passage:` for audio | `[PAPER][CODE]` conflict; public-code runs must not be labelled strict paper-protocol reproductions |
| Pooling | attention-mask-aware mean of last hidden states | `[CODE]` `OmniEmbedAdapter._encode` |
| Normalization | L2 | `[CODE]` `OmniEmbedAdapter._encode` |
| Projection | none; retain backbone hidden dimension | `[CODE]` base adapter output |

The exact immutable resources and this protocol are centralized in `configs/checkpoints/vanilla_backbones.json`. The registry deliberately records the unresolved paper/code audio-prefix conflict rather than silently choosing a value and claiming paper equivalence.

## CPU base-lock pipeline

`scripts/run_vanilla_backbone_model_pipeline.sh` performs two non-destructive stages:

1. query the pinned Hugging Face revision and compare every selected local file with the remote size and Git-LFS SHA256 or Git blob ID;
2. write a small portable lock under `results/model_locks/<backbone_id>.json`.

The pipeline disables GPU visibility, refuses dirty Git state, refuses to overwrite an existing lock, and retains a failed audit under `logs/`. It does not download files, load a model, generate embeddings, or produce retrieval metrics. Because hashing a 3B/7B snapshot may exceed 30 minutes on shared storage, each real execution must receive a separate resource/time report before it is started.

## Lock-bound embedding path

The formal generator and wrapper are committed. Nemotron-3B has completed its
real lock-bound smoke, full Clotho generation, and four-protocol finalization;
Qwen2.5-Omni-3B has a committed base-only lock inherited byte-for-byte from the
independently audited base inventory in `results/model_locks/oea_qwen3b.json`.
Its GPU smoke is the next pending gate:

```bash
bash scripts/run_vanilla_backbone_embeddings.sh vanilla_nemotron_3b --smoke
bash scripts/run_vanilla_backbone_embeddings.sh vanilla_qwen2_5_omni_3b --smoke
bash scripts/run_vanilla_backbone_embeddings.sh vanilla_qwen2_5_omni_7b --smoke
```

Before loading a GPU, the wrapper requires a clean worktree and a committed
`results/model_locks/<backbone_id>.json`. The resolver binds the protocol file,
model lock, complete base inventory, and current Git commit into a non-overwriting
resolved config. The generator then:

1. runs with all Hugging Face offline flags enabled and exactly one visible GPU;
2. rejects missing, extra, changed, or symlinked base files;
3. derives the output dimension from the verified base `config.json`;
4. loads only `OmniEmbedAdapter`, with every base parameter frozen;
5. records that no OEA checkpoint, LoRA, or projection head was loaded;
6. writes resumable L2-normalized chunks plus candidate/query metadata, full
   configuration, hashes, Git identity, environment, GPU information, and
   failure evidence.

The wrapper requires an explicit `--smoke` or `--full` mode. `--smoke` uses the
five audio files bundled with the repository (5 candidates and 25 caption
queries) and is only a model-load/forward/integrity fixture. It does not replace
the separate 32--128-row dataset-pipeline smoke tests. A full pass cannot start
unless its third argument points to a completed run from the same backbone,
current Git commit, and current committed model lock. The gate rehashes all four
output artifacts, checks their shapes and metadata counts, verifies L2 normalization,
and rejects any record that loaded an OEA checkpoint, LoRA, or projection head:

```bash
bash scripts/run_vanilla_backbone_embeddings.sh \
  vanilla_nemotron_3b \
  --full \
  /path/to/smoke-run/generation_metrics.json
```

The three pinned configurations expose hidden dimensions through two verified
structures: Nemotron uses `text_config.hidden_size` (2048), while Qwen3B and
Qwen7B use `thinker_config.text_config.hidden_size` (2048 and 3584). These are
read from the locked local file rather than hard-coded as a claimed paper
parameter. The smoke gate is implemented but has not yet passed on a real GPU.

## CPU retrieval finalizer

After a full embedding run completes, the base-only evidence finalizer runs on
CPU:

```bash
bash scripts/run_vanilla_clotho_retrieval_suite.sh \
  vanilla_nemotron_3b \
  /path/to/completed-full-embedding-directory
```

The finalizer accepts only the matching committed vanilla model lock and exact
generation-protocol SHA. It rehashes all embeddings and metadata, checks the
base revision and hidden dimension, and requires the generator to record
`projection_head_loaded=false`, `lora_loaded=false`, and
`oea_checkpoint_loaded=false`. Its evaluator identity is explicitly
`base-only:<repo>@<revision>#model-lock-sha256=...`; the compatibility field
named `checkpoint` therefore cannot be mistaken for an OEA checkpoint claim.

For each backbone it emits four separate protocols: T2A with all captions,
T2A with the public-code seed-0 one-caption selection, T2T with the public-code
seed-0 selection, and an explicitly `[INFERRED]` all-caption T2T sensitivity
run. Each protocol retains embeddings, rankings, metrics, configuration, Git
identity, and file hashes. The wrapper disables GPU visibility and never
overwrites a non-complete protocol directory.

The current protocol fixes Clotho evaluation at 1,045 candidates and all 5,225
captions. Batch size 1 and seed 42 are explicitly `[INFERRED]`; the paper does
not publish vanilla evaluation values for them. Nemotron-3B and
Qwen2.5-Omni-3B now each have four committed reproduced protocol groups in
`results/audits/vanilla_nemotron_3b_clotho_main_eval_20260730.json` and
`results/audits/vanilla_qwen2_5_omni_3b_clotho_main_eval_20260730.json`.
Qwen2.5-Omni-7B still needs its base-only lock and separate GPU approval.
