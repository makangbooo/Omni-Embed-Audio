# Training Launchers

This directory contains example scripts and configurations for training audio-text retrieval models.

## Quick Start

### Baseline Training (No Checkpoint)

Train OEA from scratch on AudioCaps:

```bash
python -m AudioRetrieval.training.oea.train_omniembed_lora \
    --dataset audiocaps \
    --train-csv AudioCaps/meta_data/train.csv \
    --val-csv AudioCaps/meta_data/val.csv \
    --audio-dir AudioCaps/audiocaps_raw_audio \
    --repo-id nvidia/omni-embed-nemotron-3b \
    --device cuda \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 512 \
    --output-dir outputs/oea_baseline
```

### Resume Training

Continue training from a checkpoint:

```bash
python -m AudioRetrieval.training.oea.train_omniembed_lora \
    --dataset audiocaps \
    --train-csv AudioCaps/meta_data/train.csv \
    --val-csv AudioCaps/meta_data/val.csv \
    --audio-dir AudioCaps/audiocaps_raw_audio \
    --init-checkpoint outputs/oea_baseline/checkpoints/best.pt \
    --repo-id nvidia/omni-embed-nemotron-3b \
    --device cuda \
    --epochs 5 \
    --output-dir outputs/oea_resumed
```

### Training with Hard Negatives

Include mined hard negatives in training:

```bash
python -m AudioRetrieval.training.oea.train_omniembed_lora \
    --dataset audiocaps \
    --train-csv AudioCaps/meta_data/train.csv \
    --val-csv AudioCaps/meta_data/val.csv \
    --audio-dir AudioCaps/audiocaps_raw_audio \
    --hard-neg-json generated_data/hard_negatives/audiocaps_train_filtered.jsonl \
    --repo-id nvidia/omni-embed-nemotron-3b \
    --device cuda \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 512 \
    --output-dir outputs/oea_hardneg
```

### Multi-GPU (DDP) Training

For distributed training across multiple GPUs:

```bash
torchrun --nproc_per_node=2 \
    -m AudioRetrieval.training.oea.train_omniembed_lora \
    --dataset audiocaps \
    --train-csv AudioCaps/meta_data/train.csv \
    --val-csv AudioCaps/meta_data/val.csv \
    --audio-dir AudioCaps/audiocaps_raw_audio \
    --repo-id nvidia/omni-embed-nemotron-3b \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 256 \
    --output-dir outputs/oea_ddp
```

### Two-Stage Training (Stage 2: Fine-tuning)

Stage 2 fine-tuning on Clotho after AudioCaps pre-training:

```bash
python -m AudioRetrieval.training.oea.train_omniembed_lora \
    --dataset clotho \
    --train-csv Clotho2.0/development_meta/clotho_captions_development.csv \
    --val-csv Clotho2.0/evaluation_meta/clotho_captions_evaluation.csv \
    --audio-dir Clotho2.0/development_audio \
    --val-audio-dir Clotho2.0/evaluation_audio \
    --init-checkpoint outputs/oea_audiocaps/checkpoints/best.pt \
    --repo-id nvidia/omni-embed-nemotron-3b \
    --device cuda \
    --epochs 5 \
    --batch-size 2 \
    --grad-accum 256 \
    --output-dir outputs/oea_clotho_stage2
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset type (audiocaps/clotho) | audiocaps |
| `--train-csv` | Training metadata CSV | Required |
| `--val-csv` | Validation metadata CSV | Required |
| `--audio-dir` | Audio files directory | Required |
| `--repo-id` | HuggingFace model repo | nvidia/omni-embed-nemotron-3b |
| `--epochs` | Number of training epochs | 3 |
| `--batch-size` | Per-device batch size | 2 |
| `--grad-accum` | Gradient accumulation steps | 512 |
| `--lr` | Learning rate | 2e-5 |
| `--hard-neg-json` | Hard negatives JSONL file | None |
| `--init-checkpoint` | Checkpoint to resume from | None |
| `--output-dir` | Output directory | outputs/omniembed_lora |
| `--wandb` | Enable W&B logging | False |

## LoRA Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--lora-rank` | LoRA rank | 16 |
| `--lora-alpha` | LoRA alpha | 32 |
| `--lora-dropout` | LoRA dropout | 0.05 |
| `--lora-targets` | Target modules | q_proj,k_proj,v_proj,o_proj |

## Projection Head Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--projection-dim` | Projection dimension | 512 |
| `--projection-dropout` | Projection dropout | 0.1 |

## Example Configurations

See `examples/` directory for pre-configured launch scripts:
- `baseline_audiocaps.sh` - Basic AudioCaps training
- `baseline_clotho.sh` - Basic Clotho training
- `hardneg_audiocaps.sh` - Training with hard negatives
- `twostage_clotho.sh` - Two-stage fine-tuning
- `ddp_3gpu.sh` - Multi-GPU distributed training
