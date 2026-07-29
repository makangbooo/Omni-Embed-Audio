#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
RUN_ID="official_oea_model_presence_scan_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse scan directory: ${RUN_DIR}" >&2
  exit 2
fi
mkdir "${RUN_DIR}"

record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
}
trap record_wrapper_exit EXIT

git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
{
  printf 'MODEL_ROOT=%q ' "${MODEL_ROOT}"
  printf '%q\n' "$0"
} > "${RUN_DIR}/command.sh"

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

if [[ -s "${RUN_DIR}/git_status.txt" ]]; then
  echo "[ERROR] Model presence scan requires a clean Git worktree." >&2
  cat "${RUN_DIR}/git_status.txt" >&2
  exit 2
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

echo "[INFO] Experiment: official OEA model metadata-only presence scan"
echo "[INFO] Git commit: $(cat "${RUN_DIR}/git_commit.txt")"
echo "[INFO] Resource: CPU only; GPU disabled"
echo "[INFO] Workload: 9 unique assets backing 6 official OEA variants"
echo "[INFO] Estimated time: 1-3 minutes"
echo "[INFO] Operation: local stat/marker scan only; no hashing, network, or downloads"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] Result: ${RUN_DIR}/model_presence.json"
echo "[INFO] Log: ${RUN_DIR}/stdout.log"

set +e
python scripts/inspect_official_oea_model_presence.py \
  --model-root "${MODEL_ROOT}" \
  --output "${RUN_DIR}/model_presence.json"
SCAN_EXIT_CODE=$?
set -e
printf '%s\n' "${SCAN_EXIT_CODE}" > "${RUN_DIR}/scan_exit_code.txt"

echo "SCAN_EXIT_CODE=${SCAN_EXIT_CODE}"
echo "RESULT_DIRECTORY=${RUN_DIR}"
echo "METRICS_PATH=${RUN_DIR}/model_presence.json"
exit "${SCAN_EXIT_CODE}"
