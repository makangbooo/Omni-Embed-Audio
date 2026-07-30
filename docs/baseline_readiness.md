# Baseline and vanilla-backbone readiness audit

This document records what the committed repository can and cannot run before
any baseline dependency or checkpoint is installed. It does not infer an
unpublished checkpoint or treat an adapter file as a complete evaluation
pipeline.

## Scope and evidence labels

- `[PAPER]`: LAION-CLAP, Robust-CLAP, MGA-CLAP, M2D-CLAP, Nemotron-3B,
  Qwen2.5-Omni-3B, and Qwen2.5-Omni-7B are the seven non-OEA rows required by
  Tables 2–5 and 11–17.
- `[CODE]`: entrypoint and path observations below come from the committed
  repository and are checked by `scripts/audit_baseline_readiness.py`.
- `[MISSING]`: no immutable revision/checkpoint identity or complete execution
  path is present in paper plus code.
- `[INFERRED]`: formal readiness requires the same fixed candidate collection,
  an immutable model identity, a complete embedding entrypoint, and successful
  local resource verification.

The machine-readable contract is
`configs/baselines/baseline_readiness.json`. Run the read-only audit with:

```bash
python scripts/audit_baseline_readiness.py \
  --output /new/path/baseline_readiness.json
```

The output path must not exist. The audit reads source and local file metadata;
it imports no model library, downloads nothing, and uses no GPU.

## Current matrix

| Model | Committed code path | Immutable resource identity | Current result |
|---|---|---|---|
| LAION-CLAP | Adapter, baseline runner, Hydra config/loader, CLI choice | `[MISSING]`; config delegates to the installed package default | `BLOCKED` |
| Robust-CLAP | Adapter and UIQ text precomputer only | `[MISSING]`; local checkpoint/source trees are named but not identified | `BLOCKED` |
| MGA-CLAP | Adapter, runner, Hydra config/loader, CLI choice | `[MISSING]`; source tree/checkpoint are untracked | `BLOCKED` |
| M2D-CLAP | Adapter and vendored portable model only | `[MISSING]`; no checkpoint revision/SHA256 | `BLOCKED` |
| Nemotron-3B | Base-lock pipeline, fixed Clotho protocol, resumable base-only generator/wrapper, and four-protocol CPU finalizer | Base snapshot is pinned by MODEL-03; real lock not yet generated | `BLOCKED` pending committed lock and GPU fixture |
| Qwen2.5-Omni-3B | Base-lock pipeline, fixed Clotho protocol, resumable base-only generator/wrapper, and four-protocol CPU finalizer | Base snapshot is pinned by MODEL-01; real lock not yet generated | `BLOCKED` pending committed lock and GPU fixture |
| Qwen2.5-Omni-7B | Base-lock pipeline, fixed Clotho protocol, resumable base-only generator/wrapper, and four-protocol CPU finalizer | Complete 20-file/22.38-GB base-only lock is committed from the clean `f861314` remote audit | `READY` pending separate GPU approval |

No row is currently formal-ready. This is not solely a checkpoint-download
problem: Robust-CLAP and M2D-CLAP lack normal baseline entrypoints; MGA-CLAP's
argparse route does not pass the required repository/checkpoint paths; and all
three vanilla rows still lack a real committed base lock and successful GPU
fixture. The committed CPU pipeline can produce those locks and the formal
wrapper consumes them, but implementing a tool is not evidence that either
remote stage has run.

## Code conflicts preserved for later fixes

1. `[CODE]` the README demonstrates Hydra `model=... dataset=...` arguments
   through `python -m AudioRetrieval`, but that command dispatches to argparse;
   `eval_hydra.py` is a separate entrypoint.
2. `[CODE]` argparse advertises an `oea` model choice, while
   `BaselineRunner._load_adapter()` has no `oea`/`omni_embed` branch.
3. `[CODE]` Robust-CLAP assumes untracked `_linguistic_robust_clap-master` and
   `torchlibrosa` trees and applies tokenizer/STFT compatibility changes. The
   original failure, upstream commit, and exact checkpoint are not committed,
   so this cannot yet be presented as a verified minimal bug fix.
4. `[CODE]` the common `OmniEmbedAdapter` implements attention-mask-aware mean
   pooling and L2 normalization, but the exact audio-prefix behavior conflicts
   with the paper statement. The formal vanilla wrapper now fixes and labels
   the public-code audio-only/no-prefix protocol; it does not claim strict
   paper-protocol equivalence.

## Unblocking order

1. Finish the common AudioCaps/MECAT candidate collections; otherwise a model
   can be loaded but cannot be compared fairly across all paper datasets.
2. Pin each CLAP source revision, dependency environment, checkpoint file,
   byte size, and SHA256. Record access/license requirements separately.
3. For Robust-CLAP, first preserve the original load error, then assess the
   existing compatibility changes as an independent minimal patch.
4. Run the non-overwriting vanilla base-lock pipeline for each complete base
   snapshot and commit only the small locks. Then run a lock-bound GPU fixture
   before full Clotho embedding generation.
5. Run `scripts/run_vanilla_clotho_retrieval_suite.sh` on the completed full
   embedding bank. It revalidates the base-only lock, protocol SHA, dimensions,
   and no-OEA/LoRA/projection flags before producing the four canonical
   T2A/T2T protocols; do not use the legacy runner's metric output as proof.
