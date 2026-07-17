# Vanilla multimodal-backbone evaluation

The paper reports three untrained multimodal-LLM backbones as retrieval baselines: Nemotron-3B, Qwen2.5-Omni-3B, and Qwen2.5-Omni-7B. A vanilla result must not load an OEA checkpoint, LoRA weights, or either 512-dimensional projection head.

## Fixed public-code protocol

| Component | Fixed value | Evidence |
| --- | --- | --- |
| Text input | prepend `query:` | `[PAPER][CODE]` |
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

Formal embedding generation remains pending. It must consume a committed base-only lock, revalidate every base file, use the fixed protocol above, save raw hidden-dimension embeddings and rankings, and first pass a small fixture before a full dataset run.
