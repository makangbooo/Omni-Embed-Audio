#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
RUN_ID="data01_clotho_eval_download_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
MANIFEST="${ROOT_DIR}/configs/resources/data01_clotho_evaluation.json"

mkdir -p "${RUN_DIR}"

record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
}
record_signal() {
  local signal_name=$1
  local exit_code=$2
  printf '%s\n' "${signal_name}" > "${RUN_DIR}/termination_signal.txt"
  exit "${exit_code}"
}
trap record_wrapper_exit EXIT
trap 'record_signal SIGHUP 129' HUP
trap 'record_signal SIGINT 130' INT
trap 'record_signal SIGTERM 143' TERM

{
  printf 'DATA_ROOT=%q ' "${DATA_ROOT}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
df -hT "$(dirname "${DATA_ROOT}")" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available." | tee "${RUN_DIR}/stderr.log" >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fx "${ENV_NAME}" >/dev/null; then
  echo "[ERROR] Conda environment '${ENV_NAME}' does not exist." | tee "${RUN_DIR}/stderr.log" >&2
  exit 3
fi

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Data root: ${DATA_ROOT}"
echo "[INFO] Resource manifest: ${MANIFEST}"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

set +e
python scripts/download_http_assets.py \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --output "${RUN_DIR}/download_manifest.json"
DOWNLOAD_EXIT_CODE=$?
set -e
printf '%s\n' "${DOWNLOAD_EXIT_CODE}" > "${RUN_DIR}/download_exit_code.txt"

df -hT "$(dirname "${DATA_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${DATA_ROOT}" > "${RUN_DIR}/data_root_size.txt" 2>&1 || true

if [[ "${DOWNLOAD_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-01 download failed; partial files were preserved for resume." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${DOWNLOAD_EXIT_CODE}"
fi

echo "[INFO] DATA-01 download and checksum verification completed"
echo "[INFO] No archive was extracted in this step"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
