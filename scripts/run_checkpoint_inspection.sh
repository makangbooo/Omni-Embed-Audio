#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/model_cache/oea}"
CHECKPOINT="${CHECKPOINT:-${MODEL_ROOT}/OEA-Qwen3B-Cl/step_40.pt}"
RUN_ID="checkpoint_inspection_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

EXPECTED_SIZE=9466833858
EXPECTED_SHA256="d5f2648c19b07fe5b33c22873098f9b9d749dfe2fb48e5d3dedfcea827c39b5c"

mkdir -p "${RUN_DIR}"

{
  printf 'MODEL_ROOT=%q CHECKPOINT=%q ' "${MODEL_ROOT}" "${CHECKPOINT}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available." | tee "${RUN_DIR}/stderr.log" >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Checkpoint: ${CHECKPOINT}"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

python scripts/inspect_oea_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --expected-size "${EXPECTED_SIZE}" \
  --expected-sha256 "${EXPECTED_SHA256}" \
  --output "${RUN_DIR}/checkpoint_inspection.json"

python -m pip freeze > "${RUN_DIR}/requirements-freeze.txt"
echo "[INFO] Checkpoint inspection completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
