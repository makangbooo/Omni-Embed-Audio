#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || "$1" != "--execute" ]]; then
  echo "Usage: bash scripts/run_asrur_whisper_numeric_diagnostic.sh --execute" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] The Whisper numerical diagnostic must run inside tmux." >&2
  echo "[INFO] Start one with: tmux new -s asrur_whisper_numeric_diag" >&2
  exit 3
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MODELS_ROOT="${MODELS_ROOT:-/home/jg525/models}"
RECORD_ID="${ASRUR_DIAGNOSTIC_RECORD_ID:-en/fiqa:snr_0:10639}"
SOURCE_MANIFEST="${DATA_ROOT}/squtr/manifests/squtr_en_fiqa_nq_audio_query_manifest.jsonl"
MAIN_CONFIG="${ROOT_DIR}/configs/asr_uncertainty_reranking/main_experiment.json"
MODEL_RESOURCE_MANIFEST="${ROOT_DIR}/configs/asr_uncertainty_reranking/resources/models.json"
RUN_ID="asrur_whisper_numeric_diagnostic_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
START_EPOCH="$(date +%s)"

mkdir -p "${RUN_DIR}"
record_exit() {
  local exit_code=$?
  local finished_epoch
  finished_epoch="$(date +%s)"
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
  printf '%s\n' "$((finished_epoch - START_EPOCH))" > "${RUN_DIR}/elapsed_seconds.txt"
}
trap record_exit EXIT
exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "===== ASRUR WHISPER NUMERICAL DIAGNOSTIC ====="
echo "record_id=${RECORD_ID}"
echo "run_dir=${RUN_DIR}"
echo "cache_mutation=disabled"
echo "training=disabled"
date -Is

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[ERROR] Formal diagnostic requires a clean Git worktree." >&2
  git status --short --untracked-files=all >&2
  exit 4
fi
for path in "${SOURCE_MANIFEST}" "${MAIN_CONFIG}" "${MODEL_RESOURCE_MANIFEST}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing prerequisite: ${path}" >&2
    exit 4
  fi
done
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short --untracked-files=all > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python scripts/validate_single_bf16_gpu.py \
  --output "${RUN_DIR}/gpu_preflight.json"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${GPU_NAME}" != *"RTX 4090"* ]]; then
  echo "[ERROR] Approved diagnostic hardware is one RTX 4090; observed: ${GPU_NAME}" >&2
  exit 5
fi
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
  --format=csv > "${RUN_DIR}/gpu_info.txt"

python scripts/verify_asrur_model_assets.py \
  --manifest "${MODEL_RESOURCE_MANIFEST}" \
  --model-root "${MODELS_ROOT}" \
  --output "${RUN_DIR}/d2_d4_model_audit.json"

python scripts/diagnose_asrur_whisper_numeric.py \
  --config "${MAIN_CONFIG}" \
  --manifest "${SOURCE_MANIFEST}" \
  --record-id "${RECORD_ID}" \
  --device cuda:0 \
  --output "${RUN_DIR}/numeric_diagnostic.json"

nvidia-smi > "${RUN_DIR}/gpu_final.txt"
echo "[INFO] Numerical diagnostic completed"
echo "[INFO] Run directory: ${RUN_DIR}"
