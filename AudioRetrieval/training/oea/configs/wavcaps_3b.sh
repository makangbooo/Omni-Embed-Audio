#!/bin/bash
# OEA Training: Qwen2.5-Omni-3B + WavCaps
# This script trains the OEA model using Qwen2.5-Omni-3B backbone with WavCaps dataset

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

DEVICE="${DEVICE:-cuda}"

# -------------------------
# Qwen2.5-Omni-3B Model
# -------------------------
LOCAL_QWEN_DIR="ModelCheckpoint/Qwen2.5-Omni-3B"
if [[ -z "${OMNI_REPO:-}" ]]; then
  if [[ -d "${LOCAL_QWEN_DIR}" && -n "$(ls -A "${LOCAL_QWEN_DIR}")" ]]; then
    OMNI_REPO="${LOCAL_QWEN_DIR}"
    echo "[INFO] Using local Qwen 3B checkpoint at ${OMNI_REPO}"
  else
    OMNI_REPO="Qwen/Qwen2.5-Omni-3B"
    echo "[WARN] Local checkpoint not found at ${LOCAL_QWEN_DIR}. Falling back to Hugging Face repo."
  fi
fi

# -------------------------
# WavCaps dataset paths
# -------------------------
WAVCAPS_AUDIO_DIR="${WAVCAPS_AUDIO_DIR:-WavCaps/audio}"
WAVCAPS_TRAIN_CSV="${WAVCAPS_TRAIN_CSV:-WavCaps/manifests/wavcaps_train.csv}"
WAVCAPS_VAL_CSV="${WAVCAPS_VAL_CSV:-WavCaps/manifests/wavcaps_val.csv}"

# ------------------------
# Training parameters
# ------------------------
BATCH_SIZE="${BATCH_SIZE:-6}"
GRAD_ACCUM="${GRAD_ACCUM:-256}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SAVE_EVERY="${SAVE_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-15}"
EARLY_STOP="${EARLY_STOP:-5}"
EPOCHS="${EPOCHS:-15}"

# ------------------------
# Checkpoints
# ------------------------
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/omniembed_lora_3b/wavcaps_pretrain}"

# ------------------------
# WandB logging
# ------------------------
WANDB_PROJECT="${WANDB_PROJECT:-omni-embed-audio-retrieval}"
WANDB_ENTITY="${WANDB_ENTITY:-jude-jiwoo-sogang-university}"
WANDB_GROUP="${WANDB_GROUP:-oea_3b_wavcaps}"
WANDB_TAGS="${WANDB_TAGS:-oea,wavcaps,3b}"
RUN_NAME="${RUN_NAME:-oea_3b_wavcaps_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "OEA Training: Qwen2.5-Omni-3B + WavCaps"
echo "============================================================"
echo "Model:"
echo "  Repo/Path     : ${OMNI_REPO}"
echo ""
echo "Dataset: WavCaps"
echo "  Train CSV     : ${WAVCAPS_TRAIN_CSV}"
echo "  Val CSV       : ${WAVCAPS_VAL_CSV}"
echo "  Audio dir     : ${WAVCAPS_AUDIO_DIR}"
echo "  Output dir    : ${OUTPUT_DIR}"
echo "  Init ckpt     : ${INIT_CHECKPOINT:-<none>}"
echo ""
echo "Hyper-parameters:"
echo "  Epochs        : ${EPOCHS}"
echo "  Batch size    : ${BATCH_SIZE}"
echo "  Grad accum    : ${GRAD_ACCUM}"
echo "  Learning rate : ${LEARNING_RATE}"
echo "  Weight decay  : ${WEIGHT_DECAY}"
echo "  Eval every    : ${EVAL_EVERY} steps"
echo "  Save every    : ${SAVE_EVERY} steps"
echo "  Early stop    : ${EARLY_STOP} evals"
echo ""
echo "WandB:"
echo "  Project       : ${WANDB_PROJECT}"
echo "  Entity        : ${WANDB_ENTITY}"
echo "  Group         : ${WANDB_GROUP}"
echo "============================================================"
echo ""

bash AudioRetrieval/training/oea/run_training.sh \
  --dataset wavcaps \
  --train-csv "${WAVCAPS_TRAIN_CSV}" \
  --val-csv "${WAVCAPS_VAL_CSV}" \
  --audio-dir "${WAVCAPS_AUDIO_DIR}" \
  --repo-id "${OMNI_REPO}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --epochs "${EPOCHS}" \
  --eval-every "${EVAL_EVERY}" \
  --save-every-steps "${SAVE_EVERY}" \
  --early-stop-patience "${EARLY_STOP}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${RUN_NAME}" \
  --wandb-tags "${WANDB_TAGS}" \
  ${INIT_CHECKPOINT:+--init-checkpoint "${INIT_CHECKPOINT}"}

echo ""
echo "[INFO] Training run finished. Checkpoints saved to ${OUTPUT_DIR}"
echo ""
