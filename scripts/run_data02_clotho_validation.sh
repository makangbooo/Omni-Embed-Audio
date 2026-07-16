#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
DATASET_ROOT="${DATA_ROOT}/clotho_v2.1"
SOURCE_ROOT="${DATASET_ROOT}/source"
ARCHIVE="${SOURCE_ROOT}/clotho_audio_evaluation.7z"
CAPTIONS="${SOURCE_ROOT}/clotho_captions_evaluation.csv"
METADATA="${SOURCE_ROOT}/clotho_metadata_evaluation.csv"
EXTRACT_ROOT="${DATASET_ROOT}/extracted"
MANIFEST_ROOT="${DATASET_ROOT}/manifests"
RUN_ID="data02_clotho_validation_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
STAGING_ROOT="${DATASET_ROOT}/.${RUN_ID}_extracting"

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
df -hT "${DATASET_ROOT}" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Dataset root: ${DATASET_ROOT}"
echo "[INFO] GPU disabled"

EXPECTED_ARCHIVE_MD5="4569624ccadf96223f19cb59fe4f849f"
ACTUAL_ARCHIVE_MD5="$(md5sum "${ARCHIVE}" | awk '{print $1}')"
printf '%s  %s\n' "${ACTUAL_ARCHIVE_MD5}" "${ARCHIVE}" > "${RUN_DIR}/archive_md5.txt"
if [[ "${ACTUAL_ARCHIVE_MD5}" != "${EXPECTED_ARCHIVE_MD5}" ]]; then
  echo "[ERROR] Archive MD5 mismatch: ${ACTUAL_ARCHIVE_MD5}" >&2
  exit 4
fi

EXTRACTOR=""
for CANDIDATE in 7zz 7z 7za; do
  if command -v "${CANDIDATE}" >/dev/null 2>&1; then
    EXTRACTOR="$(command -v "${CANDIDATE}")"
    break
  fi
done
if [[ -z "${EXTRACTOR}" ]]; then
  echo "[ERROR] No 7z-compatible extractor is available (tried 7zz, 7z, 7za)." >&2
  exit 5
fi
"${EXTRACTOR}" i > "${RUN_DIR}/extractor_info.txt" 2>&1 || true
echo "[INFO] Extractor: ${EXTRACTOR}"

if [[ -d "${EXTRACT_ROOT}" ]]; then
  if [[ ! -f "${EXTRACT_ROOT}/.data02_extraction_complete" ]]; then
    echo "[ERROR] Existing extract root has no completion marker; refusing to overwrite:" >&2
    echo "[ERROR] ${EXTRACT_ROOT}" >&2
    exit 6
  fi
  echo "[INFO] Reusing previously completed extraction: ${EXTRACT_ROOT}"
else
  if [[ -e "${STAGING_ROOT}" ]]; then
    echo "[ERROR] Staging path already exists; refusing to overwrite: ${STAGING_ROOT}" >&2
    exit 7
  fi
  mkdir -p "${STAGING_ROOT}"
  echo "[INFO] Extracting into isolated staging directory: ${STAGING_ROOT}"
  set +e
  "${EXTRACTOR}" x "${ARCHIVE}" "-o${STAGING_ROOT}" -bso1 -bsp1 -bse1
  EXTRACTION_EXIT_CODE=$?
  set -e
  printf '%s\n' "${EXTRACTION_EXIT_CODE}" > "${RUN_DIR}/extraction_exit_code.txt"
  if [[ "${EXTRACTION_EXIT_CODE}" -ne 0 ]]; then
    echo "[ERROR] Extraction failed; staging directory was preserved for audit." >&2
    exit "${EXTRACTION_EXIT_CODE}"
  fi
  WAV_COUNT="$(find "${STAGING_ROOT}" -type f -iname '*.wav' | wc -l)"
  printf '%s\n' "${WAV_COUNT}" > "${RUN_DIR}/staging_wav_count.txt"
  if [[ "${WAV_COUNT}" -ne 1045 ]]; then
    echo "[ERROR] Extracted WAV count mismatch: ${WAV_COUNT} != 1045" >&2
    exit 8
  fi
  mv "${STAGING_ROOT}" "${EXTRACT_ROOT}"
  {
    printf 'archive_md5=%s\n' "${ACTUAL_ARCHIVE_MD5}"
    printf 'git_commit=%s\n' "$(cat "${RUN_DIR}/git_commit.txt")"
    printf 'completed_at=%s\n' "$(date -Is)"
  } > "${EXTRACT_ROOT}/.data02_extraction_complete"
fi

mkdir -p "${MANIFEST_ROOT}"
set +e
python scripts/validate_clotho_evaluation.py \
  --extract-root "${EXTRACT_ROOT}" \
  --captions-csv "${CAPTIONS}" \
  --metadata-csv "${METADATA}" \
  --uiq-root "${ROOT_DIR}/data/UIQ/clotho" \
  --manifest-output "${MANIFEST_ROOT}/clotho_evaluation_manifest.jsonl" \
  --statistics-output "${RUN_DIR}/data_statistics.json" \
  --expected-examples 1045 \
  --expected-negative-rows 542
VALIDATION_EXIT_CODE=$?
set -e
printf '%s\n' "${VALIDATION_EXIT_CODE}" > "${RUN_DIR}/validation_exit_code.txt"
if [[ "${VALIDATION_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-02 validation failed; extracted files and audit evidence were preserved." >&2
  exit "${VALIDATION_EXIT_CODE}"
fi

find "${EXTRACT_ROOT}" -type f -iname '*.wav' -printf '%s %p\n' \
  | sort > "${RUN_DIR}/audio_files.txt"
md5sum "${MANIFEST_ROOT}/clotho_evaluation_manifest.jsonl" \
  > "${RUN_DIR}/manifest_md5.txt"
df -hT "${DATASET_ROOT}" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${DATASET_ROOT}" > "${RUN_DIR}/dataset_size.txt"

echo "[INFO] DATA-02 extraction and validation completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
echo "[INFO] Canonical manifest: ${MANIFEST_ROOT}/clotho_evaluation_manifest.jsonl"
