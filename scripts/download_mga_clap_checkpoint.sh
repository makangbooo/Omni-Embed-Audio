#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
OVERLAY="${MODEL_ROOT}/python/paper-model-download-tools-2"
TOOLS_REQUIREMENTS="${ROOT_DIR}/configs/resources/paper_model_download_tools.requirements.txt"
DESTINATION="${MODEL_ROOT}/mga-clap/pretrained_models/models/model.pt"
PARTIAL="${DESTINATION}.partial"
LOCK_FILE="${DESTINATION}.download.lock"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/mga_clap_checkpoint_${STAMP}"
LOG_FILE="${RUN_DIR}/combined.log"

mkdir -p "${RUN_DIR}" "$(dirname "${DESTINATION}")"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

finish() {
  local rc="$1" status="$2"
  printf '%s\n' "${rc}" > "${RUN_DIR}/exit_code.txt"
  echo "FINAL_RUN_RC=${rc}"
  echo "COMPLETION_STATUS=${status}"
  echo "RESULT_DIRECTORY=${RUN_DIR}"
  echo "CHECKPOINT_PATH=${DESTINATION}"
  exit "${rc}"
}

echo "EXPERIMENT_NAME=MGA-CLAP official checkpoint download"
echo "GPU_USED=no"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "ESTIMATED_TOTAL_TIME=5-30 minutes"
echo "LOG_FILE=${LOG_FILE}"

if [[ -s "${DESTINATION}" ]]; then
  echo "MGA_CHECKPOINT_STATUS=reused"
  if ! stat -c 'CHECKPOINT_BYTES=%s' "${DESTINATION}"; then
    finish 4 failed_existing_checkpoint_stat
  fi
  if ! sha256sum "${DESTINATION}"; then
    finish 4 failed_existing_checkpoint_hash
  fi
  finish 0 complete
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "[ERROR] flock is required for cross-server download locking" >&2
  finish 2 failed_missing_flock
fi
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[ERROR] Another MGA checkpoint download is already running" >&2
  finish 2 failed_download_already_running
fi
CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES=""

EXPECTED_TOOLS_SHA="$(sha256sum "${TOOLS_REQUIREMENTS}" | awk '{print $1}')"
TOOLS_MARKER="${OVERLAY}/.oea_requirements_sha256"
if [[ -d "${OVERLAY}/gdown" ]] \
   && [[ -f "${TOOLS_MARKER}" ]] \
   && [[ "$(<"${TOOLS_MARKER}")" == "${EXPECTED_TOOLS_SHA}" ]]; then
  echo "DOWNLOAD_TOOLS_STATUS=reused"
elif [[ -e "${OVERLAY}" ]]; then
  echo "[ERROR] Existing download-tool overlay has an unexpected identity: ${OVERLAY}" >&2
  finish 2 failed_download_tools_identity
else
  TOOLS_ATTEMPT="${OVERLAY}.attempt-${STAMP}"
  mkdir -p "$(dirname "${TOOLS_ATTEMPT}")"
  python -m pip install --no-deps --require-hashes \
    --target "${TOOLS_ATTEMPT}" --requirement "${TOOLS_REQUIREMENTS}"
  RC=$?
  if [[ "${RC}" -ne 0 ]]; then
    finish "${RC}" failed_download_tools_install
  fi
  printf '%s\n' "${EXPECTED_TOOLS_SHA}" > "${TOOLS_ATTEMPT}/.oea_requirements_sha256"
  if ! mv "${TOOLS_ATTEMPT}" "${OVERLAY}"; then
    finish 2 failed_download_tools_finalize
  fi
  echo "DOWNLOAD_TOOLS_STATUS=installed"
fi

PYTHONPATH="${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m gdown --continue 1RWTuVMEPy-L0uK6WYIX2wwxHjD1YSQFz \
    --output "${PARTIAL}"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed_download
fi
if [[ ! -s "${PARTIAL}" ]]; then
  finish 3 failed_empty_download
fi
python - "${PARTIAL}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
prefix = path.read_bytes()[:256].lower()
if path.stat().st_size < 1024 * 1024:
    raise SystemExit("checkpoint is unexpectedly smaller than 1 MiB")
if b"<html" in prefix or b"<!doctype" in prefix:
    raise SystemExit("downloaded file is HTML, not a checkpoint")
PY
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed_content_gate
fi
if ! mv "${PARTIAL}" "${DESTINATION}"; then
  finish 4 failed_finalize
fi
echo "MGA_CHECKPOINT_STATUS=complete"
if ! stat -c 'CHECKPOINT_BYTES=%s' "${DESTINATION}"; then
  finish 4 failed_checkpoint_stat
fi
if ! sha256sum "${DESTINATION}" | tee "${RUN_DIR}/checkpoint_sha256.txt"; then
  finish 4 failed_checkpoint_hash
fi
finish 0 complete
