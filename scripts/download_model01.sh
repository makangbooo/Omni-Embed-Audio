#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/model_cache/oea}"
RUN_ID="model01_download_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
MANIFEST="${ROOT_DIR}/configs/resources/model01_qwen3b_cl.json"

mkdir -p "${RUN_DIR}"

{
  printf 'MODEL_ROOT=%q ' "${MODEL_ROOT}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

if ! resolve_conda_executable >/dev/null 2>&1; then
  echo "[ERROR] conda is not available." | tee "${RUN_DIR}/stderr.log" >&2
  exit 2
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fx "${ENV_NAME}" >/dev/null; then
  echo "[ERROR] Conda environment '${ENV_NAME}' does not exist." | tee "${RUN_DIR}/stderr.log" >&2
  exit 3
fi

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] Resource manifest: ${MANIFEST}"
conda activate "${ENV_NAME}"

python scripts/download_model_assets.py \
  --manifest "${MANIFEST}" \
  --model-root "${MODEL_ROOT}" \
  --output "${RUN_DIR}/download_manifest.json"

df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${MODEL_ROOT}" > "${RUN_DIR}/model_root_size.txt"

echo "[INFO] MODEL-01 download completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
