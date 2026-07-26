#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
DATASET_ROOT="${DATA_ROOT}/squtr"
ARCHIVE="${DATASET_ROOT}/source/source_data.zip"
EXTRACT_ROOT="${DATASET_ROOT}/extracted"
MANIFEST_ROOT="${DATASET_ROOT}/manifests"
RESOURCE_MANIFEST="${ROOT_DIR}/configs/resources/data12_squtr.json"
STRUCTURE_MANIFEST="${ROOT_DIR}/configs/resources/data13_squtr_structure.json"
MANIFEST_OUTPUT="${MANIFEST_ROOT}/squtr_audio_query_manifest.jsonl"
RUN_ID="data13b_squtr_validation_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
MIN_FREE_BYTES="${MIN_FREE_BYTES:-40000000000}"

mkdir -p "${RUN_DIR}"
record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
}
trap record_wrapper_exit EXIT

{
  printf 'DATA_ROOT=%q ' "${DATA_ROOT}"
  printf 'MIN_FREE_BYTES=%q ' "${MIN_FREE_BYTES}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short --untracked-files=all > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
df -hT "${DATASET_ROOT}" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)
echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Archive: ${ARCHIVE}"
echo "[INFO] Extract root: ${EXTRACT_ROOT}"
echo "[INFO] Manifest: ${MANIFEST_OUTPUT}"
echo "[INFO] GPU disabled"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "[ERROR] Missing SQuTR archive: ${ARCHIVE}" >&2
  exit 2
fi
if [[ -L "${EXTRACT_ROOT}" ]]; then
  echo "[ERROR] Extract root must not be a symbolic link: ${EXTRACT_ROOT}" >&2
  exit 3
fi

mkdir -p "${DATASET_ROOT}" "${MANIFEST_ROOT}"
if [[ ! -f "${EXTRACT_ROOT}/.data13b_extraction_complete.json" ]]; then
  AVAILABLE_BYTES="$(
    df --output=avail -B1 "${DATASET_ROOT}" | tail -n 1 | tr -d '[:space:]'
  )"
  if [[ ! "${AVAILABLE_BYTES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Unable to determine available disk bytes." >&2
    exit 4
  fi
  if (( AVAILABLE_BYTES < MIN_FREE_BYTES )); then
    echo "[ERROR] Insufficient free disk for audited extraction." >&2
    echo "[ERROR] available=${AVAILABLE_BYTES}, required=${MIN_FREE_BYTES}" >&2
    exit 5
  fi
fi

set +e
python scripts/extract_squtr_archive.py \
  --archive "${ARCHIVE}" \
  --extract-root "${EXTRACT_ROOT}" \
  --resource-manifest "${RESOURCE_MANIFEST}" \
  --structure-manifest "${STRUCTURE_MANIFEST}" \
  --output "${RUN_DIR}/extraction_report.json"
EXTRACTION_EXIT_CODE=$?
set -e
printf '%s\n' "${EXTRACTION_EXIT_CODE}" > "${RUN_DIR}/extraction_exit_code.txt"

if [[ "${EXTRACTION_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-13B extraction failed; partial progress was preserved." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${EXTRACTION_EXIT_CODE}"
fi

set +e
python scripts/validate_squtr_extracted.py \
  --extract-root "${EXTRACT_ROOT}" \
  --structure-manifest "${STRUCTURE_MANIFEST}" \
  --manifest-output "${MANIFEST_OUTPUT}" \
  --statistics-output "${RUN_DIR}/data_statistics.json"
VALIDATION_EXIT_CODE=$?
set -e
printf '%s\n' "${VALIDATION_EXIT_CODE}" > "${RUN_DIR}/validation_exit_code.txt"

df -hT "${DATASET_ROOT}" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${EXTRACT_ROOT}" "${MANIFEST_OUTPUT}" \
  > "${RUN_DIR}/artifact_sizes.txt" 2>&1 || true
if [[ -f "${MANIFEST_OUTPUT}" ]]; then
  sha256sum "${MANIFEST_OUTPUT}" > "${RUN_DIR}/manifest_sha256.txt"
fi

echo "===== EXTRACTION REPORT ====="
cat "${RUN_DIR}/extraction_report.json"
echo "===== DATA STATISTICS ====="
cat "${RUN_DIR}/data_statistics.json"
echo "===== ARTIFACT SIZES ====="
cat "${RUN_DIR}/artifact_sizes.txt"
if [[ -f "${RUN_DIR}/manifest_sha256.txt" ]]; then
  echo "===== MANIFEST SHA256 ====="
  cat "${RUN_DIR}/manifest_sha256.txt"
fi

if [[ "${VALIDATION_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-13B content validation failed; evidence was preserved." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${VALIDATION_EXIT_CODE}"
fi

echo "[INFO] DATA-13B extraction and content validation completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
