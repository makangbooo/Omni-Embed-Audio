#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MAX_ATTEMPTS="${DATA_DOWNLOAD_MAX_ATTEMPTS:-20}"
BACKOFF_SECONDS="${DATA_DOWNLOAD_RETRY_BACKOFF_SECONDS:-30}"
MAX_RETRY_DELAY_SECONDS="${DATA_DOWNLOAD_MAX_RETRY_DELAY_SECONDS:-600}"
MAX_WORKERS="${DATA_DOWNLOAD_MAX_WORKERS:-1}"
RUN_ID="data08_wavcaps_metadata_download_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
MANIFEST="${ROOT_DIR}/configs/resources/data08_wavcaps_metadata.json"

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

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)
echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Data root: ${DATA_ROOT}"
echo "[INFO] GPU disabled"
echo "[INFO] Download scope: 8 pinned metadata/blacklist files (176,863,095 bytes)"
echo "[INFO] Audio archives under Zip_files/** are excluded"

set +e
python scripts/download_model_assets.py \
  --manifest "${MANIFEST}" \
  --resource-root "${DATA_ROOT}" \
  --output "${RUN_DIR}/download_manifest.json" \
  --max-download-attempts "${MAX_ATTEMPTS}" \
  --retry-backoff-seconds "${BACKOFF_SECONDS}" \
  --max-retry-delay-seconds "${MAX_RETRY_DELAY_SECONDS}" \
  --max-workers "${MAX_WORKERS}"
DOWNLOAD_EXIT_CODE=$?
set -e
printf '%s\n' "${DOWNLOAD_EXIT_CODE}" > "${RUN_DIR}/download_exit_code.txt"

df -hT "$(dirname "${DATA_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${DATA_ROOT}/wavcaps_0930ec11" \
  > "${RUN_DIR}/dataset_size.txt" 2>&1 || true
if [[ "${DOWNLOAD_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-08 download failed; partial files were preserved for resume." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${DOWNLOAD_EXIT_CODE}"
fi

echo "[INFO] DATA-08 WavCaps metadata download completed"
echo "[INFO] Exact bytes and local SHA256 verified for all 8 files"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
