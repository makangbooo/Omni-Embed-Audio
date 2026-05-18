#!/bin/bash
# OEA training with hard negatives on AudioCaps
# Usage: ./hardneg_audiocaps.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

# Ensure hard negatives file exists
HARD_NEG_FILE="generated_data/hard_negatives/audiocaps_train_filtered.jsonl"
if [ ! -f "$HARD_NEG_FILE" ]; then
    echo "Hard negatives file not found: $HARD_NEG_FILE"
    echo "Please run hard negative mining first:"
    echo "  python -m AudioRetrieval preprocess hard-negatives ..."
    exit 1
fi

python -m AudioRetrieval.training.oea.train_omniembed_lora \
    --dataset audiocaps \
    --train-csv AudioCaps/meta_data/train.csv \
    --val-csv AudioCaps/meta_data/val.csv \
    --audio-dir AudioCaps/audiocaps_raw_audio \
    --hard-neg-json "$HARD_NEG_FILE" \
    --repo-id nvidia/omni-embed-nemotron-3b \
    --device cuda \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 512 \
    --lr 2e-5 \
    --output-dir outputs/oea_audiocaps_hardneg \
    "$@"
