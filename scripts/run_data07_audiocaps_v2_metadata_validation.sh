#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
DATASET_ROOT="${DATA_ROOT}/audiocaps_v2_d004db3"
METADATA_ROOT="${DATASET_ROOT}/metadata"
MANIFEST_ROOT="${DATASET_ROOT}/manifests"
RUN_ID="data07_audiocaps_v2_metadata_validation_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

mkdir -p "${RUN_DIR}" "${MANIFEST_ROOT}"
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
df -hT "${DATASET_ROOT}" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""
exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Metadata root: ${METADATA_ROOT}"
echo "[INFO] GPU disabled"

set +e
python scripts/validate_audiocaps_v2_metadata.py \
  --metadata-root "${METADATA_ROOT}" \
  --uiq-root "${ROOT_DIR}/data/UIQ/audiocaps" \
  --manifest-root "${MANIFEST_ROOT}" \
  --statistics-output "${RUN_DIR}/data_statistics.json"
VALIDATION_EXIT_CODE=$?
set -e
printf '%s\n' "${VALIDATION_EXIT_CODE}" > "${RUN_DIR}/validation_exit_code.txt"

df -hT "${DATASET_ROOT}" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${DATASET_ROOT}" > "${RUN_DIR}/dataset_size.txt" 2>&1 || true
if [[ "${VALIDATION_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-07 validation failed; audit evidence was preserved." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${VALIDATION_EXIT_CODE}"
fi

sha256sum "${MANIFEST_ROOT}"/*.jsonl > "${RUN_DIR}/manifest_sha256.txt"
echo "[INFO] DATA-07 AudioCaps 2.0 metadata/UIQ validation completed"
echo "[WARN] Paper/README report 91,256 train rows; public OEA loader yields 91,254"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
