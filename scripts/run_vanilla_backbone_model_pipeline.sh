#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <backbone_id>" >&2
  exit 2
fi

BACKBONE_ID=$1
if [[ ! "${BACKBONE_ID}" =~ ^[a-z0-9]+(_[a-z0-9]+)*$ ]]; then
  echo "[ERROR] Invalid backbone ID: ${BACKBONE_ID}" >&2
  exit 2
fi

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
REGISTRY="${ROOT_DIR}/configs/checkpoints/vanilla_backbones.json"
MODEL_LOCK="${ROOT_DIR}/results/model_locks/${BACKBONE_ID}.json"
RUN_ID="vanilla_model_pipeline_${BACKBONE_ID}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
PIPELINE_DIR="${RUN_DIR}/pipeline"

if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse pipeline run directory: ${RUN_DIR}" >&2
  exit 2
fi
mkdir "${RUN_DIR}"

record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
}
trap record_wrapper_exit EXIT

{
  printf 'MODEL_ROOT=%q ' "${MODEL_ROOT}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
free -h > "${RUN_DIR}/memory_before.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

if [[ -s "${RUN_DIR}/git_status.txt" ]]; then
  echo "[ERROR] Vanilla model pipeline requires a clean Git worktree." >&2
  cat "${RUN_DIR}/git_status.txt" >&2
  exit 2
fi
if [[ -e "${MODEL_LOCK}" ]]; then
  echo "[ERROR] Refusing to overwrite existing model lock: ${MODEL_LOCK}" >&2
  exit 2
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Backbone: ${BACKBONE_ID}"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] Model lock: ${MODEL_LOCK}"
echo "[INFO] GPU disabled"
echo "[INFO] Stages: exact base resource audit -> vanilla base-only model lock"

set +e
python scripts/run_vanilla_backbone_model_pipeline.py \
  --registry "${REGISTRY}" \
  --backbone "${BACKBONE_ID}" \
  --model-root "${MODEL_ROOT}" \
  --output-dir "${PIPELINE_DIR}" \
  --lock-output "${MODEL_LOCK}"
PIPELINE_EXIT_CODE=$?
set -e
printf '%s\n' "${PIPELINE_EXIT_CODE}" > "${RUN_DIR}/pipeline_exit_code.txt"

free -h > "${RUN_DIR}/memory_after.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
python -m json.tool "${PIPELINE_DIR}/pipeline_manifest.json" || true

if [[ "${PIPELINE_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] Vanilla model pipeline failed; audit evidence was retained." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${PIPELINE_EXIT_CODE}"
fi

echo "[INFO] Vanilla base-only model pipeline completed"
echo "[INFO] Review and commit the small model lock: ${MODEL_LOCK}"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
