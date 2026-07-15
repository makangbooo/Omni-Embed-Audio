#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="oea-repro"
RUN_ID="environment_gpu_validation_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

mkdir -p "${RUN_DIR}"

{
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
nvidia-smi > "${RUN_DIR}/gpu_info.txt" 2>&1

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available." | tee "${RUN_DIR}/stderr.log" >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[ERROR] Conda environment '${ENV_NAME}' does not exist." | tee "${RUN_DIR}/stderr.log" >&2
  exit 3
fi

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Validating ${ENV_NAME} on visible GPU devices"
conda activate "${ENV_NAME}"

python -m pip check
python scripts/check_environment.py \
  --mode gpu \
  --output "${RUN_DIR}/environment_check_gpu.json"

echo "[INFO] GPU environment validation completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
