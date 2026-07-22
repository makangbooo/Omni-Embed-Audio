#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
CURL_RETRIES="${DATA_DOWNLOAD_CURL_RETRIES:-30}"
MIN_FREE_BYTES="${SQUTR_MIN_FREE_BYTES:-32212254720}"
RUN_ID="data12_squtr_download_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
MANIFEST="${ROOT_DIR}/configs/resources/data12_squtr.json"
DATASET_DIR="${DATA_ROOT}/squtr"
ARCHIVE="${DATASET_DIR}/source/source_data.zip"

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

AVAILABLE_BYTES="$(df --output=avail -B1 "$(dirname "${DATA_ROOT}")" | tail -n 1 | tr -d ' ')"
if [[ ! "${AVAILABLE_BYTES}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] Could not determine available bytes for ${DATA_ROOT}" >&2
  exit 4
fi
if (( AVAILABLE_BYTES < MIN_FREE_BYTES )); then
  echo "[ERROR] DATA-12 requires at least ${MIN_FREE_BYTES} free bytes; found ${AVAILABLE_BYTES}" >&2
  exit 5
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)
echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Data root: ${DATA_ROOT}"
echo "[INFO] GPU disabled"
echo "[INFO] Pinned SQuTR revision: 2f1b041e2e98e0d28ed68fbcf22126ef247eb719"
echo "[INFO] Archive: ${ARCHIVE}"
echo "[INFO] Expected bytes: 21069841248"
echo "[INFO] This step downloads and verifies only; it does not extract the ZIP archive"

set +e
python scripts/download_http_assets.py \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --output "${RUN_DIR}/download_manifest.json" \
  --curl-retries "${CURL_RETRIES}"
DOWNLOAD_EXIT_CODE=$?
set -e
printf '%s\n' "${DOWNLOAD_EXIT_CODE}" > "${RUN_DIR}/download_exit_code.txt"

df -hT "$(dirname "${DATA_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${DATASET_DIR}" > "${RUN_DIR}/dataset_size.txt" 2>&1 || true

if [[ "${DOWNLOAD_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-12 download failed; the .part file was preserved for resume." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${DOWNLOAD_EXIT_CODE}"
fi

ls -lh "${ARCHIVE}" | tee "${RUN_DIR}/archive_listing.txt"
sha256sum "${ARCHIVE}" | tee "${RUN_DIR}/archive_sha256.txt"
echo "[INFO] DATA-12 SQuTR download and SHA256 verification completed"
echo "[INFO] No archive was extracted in this step"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
