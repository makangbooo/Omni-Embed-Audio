#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
DATASET_ROOT="${DATA_ROOT}/wavcaps_0930ec11"
WAVCAPS_ROOT="${DATASET_ROOT}/source"
AUDIOCAPS_TEST_CSV="${DATA_ROOT}/audiocaps_v2_d004db3/metadata/test.csv"
CLOTHO_METADATA_CSV="${DATA_ROOT}/clotho_v2.1/source/clotho_metadata_evaluation.csv"
MANIFEST_ROOT="${DATASET_ROOT}/manifests"
BLOCKLIST_ROOT="${DATASET_ROOT}/blocklists"
RUN_ID="data09_wavcaps_metadata_audit_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

mkdir -p "${RUN_DIR}" "${MANIFEST_ROOT}" "${BLOCKLIST_ROOT}"
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
echo "[INFO] WavCaps metadata root: ${WAVCAPS_ROOT}"
echo "[INFO] AudioCaps test CSV: ${AUDIOCAPS_TEST_CSV}"
echo "[INFO] Clotho metadata CSV: ${CLOTHO_METADATA_CSV}"
echo "[INFO] GPU disabled"
echo "[INFO] No audio archives are read or downloaded in DATA-09"

set +e
python scripts/validate_wavcaps_metadata.py \
  --wavcaps-root "${WAVCAPS_ROOT}" \
  --audiocaps-test-csv "${AUDIOCAPS_TEST_CSV}" \
  --clotho-metadata-csv "${CLOTHO_METADATA_CSV}" \
  --manifest-root "${MANIFEST_ROOT}" \
  --blocklist-root "${BLOCKLIST_ROOT}" \
  --statistics-output "${RUN_DIR}/data_statistics.json"
VALIDATION_EXIT_CODE=$?
set -e
printf '%s\n' "${VALIDATION_EXIT_CODE}" > "${RUN_DIR}/validation_exit_code.txt"

df -hT "${DATASET_ROOT}" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${DATASET_ROOT}" > "${RUN_DIR}/dataset_size.txt" 2>&1 || true
if [[ "${VALIDATION_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-09 validation failed; evidence was preserved." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${VALIDATION_EXIT_CODE}"
fi

sha256sum "${MANIFEST_ROOT}"/*.jsonl "${BLOCKLIST_ROOT}"/*.jsonl \
  > "${RUN_DIR}/artifact_sha256.txt"
echo "[INFO] DATA-09 WavCaps metadata/count/leakage audit completed"
echo "[INFO] Full 403,050-row metadata manifest: ${MANIFEST_ROOT}/wavcaps_all_metadata_audit_manifest.jsonl"
echo "[WARN] Paper says <=31 seconds; public metadata reaches 275,618 only with 0 < duration < 31"
echo "[WARN] Exact OEA post-blocklist training manifest remains unpublished"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
