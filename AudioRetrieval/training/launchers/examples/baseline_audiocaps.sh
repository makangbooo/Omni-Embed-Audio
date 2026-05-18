#!/bin/bash
# Baseline OEA training on AudioCaps
# Usage: ./baseline_audiocaps.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

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
    --lr 2e-5 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --projection-dim 512 \
    --output-dir outputs/oea_audiocaps_baseline \
    "$@"
