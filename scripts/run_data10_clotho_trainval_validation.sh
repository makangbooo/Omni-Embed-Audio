#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
DATASET_ROOT="${DATA_ROOT}/clotho_v2.1"
SOURCE_ROOT="${DATASET_ROOT}/source"
EXTRACT_ROOT="${DATASET_ROOT}/extracted_trainval"
MANIFEST_ROOT="${DATASET_ROOT}/manifests"
RUN_ID="data10_clotho_trainval_validation_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

DEVELOPMENT_ARCHIVE="${SOURCE_ROOT}/clotho_audio_development.7z"
VALIDATION_ARCHIVE="${SOURCE_ROOT}/clotho_audio_validation.7z"
DEVELOPMENT_CAPTIONS="${SOURCE_ROOT}/clotho_captions_development.csv"
VALIDATION_CAPTIONS="${SOURCE_ROOT}/clotho_captions_validation.csv"
DEVELOPMENT_METADATA="${SOURCE_ROOT}/clotho_metadata_development.csv"
VALIDATION_METADATA="${SOURCE_ROOT}/clotho_metadata_validation.csv"

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

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Dataset root: ${DATASET_ROOT}"
echo "[INFO] GPU disabled"

SOURCE_FILES=(
  "clotho_audio_development.7z"
  "clotho_audio_validation.7z"
  "clotho_captions_development.csv"
  "clotho_captions_validation.csv"
  "clotho_metadata_development.csv"
  "clotho_metadata_validation.csv"
)
EXPECTED_MD5=(
  "c8b05bc7acdb13895bb3c6a29608667e"
  "7dba730be08bada48bd15dc4e668df59"
  "d4090b39ce9f2491908eebf4d5b09bae"
  "5879e023032b22a2c930aaa0528bead4"
  "170d20935ecfdf161ce1bb154118cda5"
  "2e010427c56b1ce6008b0f03f41048ce"
)
EXPECTED_SIZE=(
  "4541582263"
  "1260701425"
  "1336762"
  "367649"
  "830797"
  "224803"
)
EXPECTED_SHA256=(
  ""
  ""
  "df2e5b92060b4bb23311f8b3a7f82241d900b9c4f62b0cc467ac2ce5e9c52886"
  "fb0365506fe2dfcba9b7299daf7623a795abbd6ab9997a88ab0308e2fdfdbb88"
  "b054a8d9d0f88436e7cf6341c82a70e9e975c3b3ddd560d9b04f2cd5fdc75949"
  "066026ae1bc20277614ae9d4fffea085d959f0b5e40120751b8d3e717f5faa97"
)

: > "${RUN_DIR}/source_md5.txt"
: > "${RUN_DIR}/source_sha256.txt"
: > "${RUN_DIR}/source_size.txt"
for index in "${!SOURCE_FILES[@]}"; do
  file="${SOURCE_ROOT}/${SOURCE_FILES[$index]}"
  expected="${EXPECTED_MD5[$index]}"
  if [[ ! -f "${file}" ]]; then
    echo "[ERROR] Required DATA-03 source file is missing: ${file}" >&2
    exit 4
  fi
  actual="$(md5sum "${file}" | awk '{print $1}')"
  printf '%s  %s\n' "${actual}" "${file}" >> "${RUN_DIR}/source_md5.txt"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[ERROR] Source MD5 mismatch for ${file}: ${actual} != ${expected}" >&2
    exit 5
  fi
  actual_size="$(stat -c '%s' "${file}")"
  printf '%s  %s\n' "${actual_size}" "${file}" >> "${RUN_DIR}/source_size.txt"
  if [[ "${actual_size}" != "${EXPECTED_SIZE[$index]}" ]]; then
    echo "[ERROR] Source size mismatch for ${file}: ${actual_size} != ${EXPECTED_SIZE[$index]}" >&2
    exit 5
  fi
  expected_sha256="${EXPECTED_SHA256[$index]}"
  if [[ -n "${expected_sha256}" ]]; then
    actual_sha256="$(sha256sum "${file}" | awk '{print $1}')"
    printf '%s  %s\n' "${actual_sha256}" "${file}" >> "${RUN_DIR}/source_sha256.txt"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
      echo "[ERROR] Source SHA256 mismatch for ${file}: ${actual_sha256} != ${expected_sha256}" >&2
      exit 5
    fi
  fi
done
echo "[INFO] All six pinned DATA-03 size/MD5 checks passed; four CSV SHA256 checks passed"

EXTRACTOR=""
for candidate in 7zz 7z 7za; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    EXTRACTOR="$(command -v "${candidate}")"
    break
  fi
done
if [[ -z "${EXTRACTOR}" ]]; then
  echo "[ERROR] No 7z-compatible extractor is available (tried 7zz, 7z, 7za)." >&2
  exit 6
fi
"${EXTRACTOR}" i > "${RUN_DIR}/extractor_info.txt" 2>&1 || true
echo "[INFO] Extractor: ${EXTRACTOR}"

mkdir -p "${EXTRACT_ROOT}" "${MANIFEST_ROOT}"

extract_split() {
  local split=$1
  local archive=$2
  local expected_md5=$3
  local expected_wav=$4
  local final_root="${EXTRACT_ROOT}/${split}"
  local staging_root="${DATASET_ROOT}/.${RUN_ID}_${split}_extracting"
  local marker="${final_root}/.data10_extraction_complete"
  local listing="${RUN_DIR}/${split}_archive_listing.txt"

  set +e
  "${EXTRACTOR}" l -slt "${archive}" > "${listing}" 2> "${RUN_DIR}/${split}_archive_listing_stderr.txt"
  local listing_exit=$?
  set -e
  printf '%s\n' "${listing_exit}" > "${RUN_DIR}/${split}_listing_exit_code.txt"
  if [[ "${listing_exit}" -ne 0 ]]; then
    echo "[ERROR] Unable to list ${split} archive safely." >&2
    exit "${listing_exit}"
  fi
  python scripts/audit_7z_listing.py \
    --listing "${listing}" \
    --archive "${archive}" \
    --expected-wav-files "${expected_wav}" \
    --output "${RUN_DIR}/${split}_archive_audit.json"

  if [[ -d "${final_root}" ]]; then
    if [[ ! -f "${marker}" ]]; then
      echo "[ERROR] Existing ${split} extraction has no completion marker:" >&2
      echo "[ERROR] ${final_root}" >&2
      exit 7
    fi
    local recorded_md5
    recorded_md5="$(sed -n 's/^archive_md5=//p' "${marker}")"
    if [[ "${recorded_md5}" != "${expected_md5}" ]]; then
      echo "[ERROR] Existing ${split} extraction marker has the wrong archive MD5." >&2
      exit 8
    fi
    echo "[INFO] Reusing previously completed ${split} extraction: ${final_root}"
    return
  fi

  if [[ -e "${staging_root}" ]]; then
    echo "[ERROR] Staging path already exists; refusing to overwrite: ${staging_root}" >&2
    exit 9
  fi
  mkdir -p "${staging_root}"
  echo "[INFO] Extracting ${split} into isolated staging directory: ${staging_root}"
  set +e
  "${EXTRACTOR}" x "${archive}" "-o${staging_root}" -bso1 -bsp1 -bse1
  local extraction_exit=$?
  set -e
  printf '%s\n' "${extraction_exit}" > "${RUN_DIR}/${split}_extraction_exit_code.txt"
  if [[ "${extraction_exit}" -ne 0 ]]; then
    echo "[ERROR] ${split} extraction failed; staging was preserved for audit." >&2
    exit "${extraction_exit}"
  fi
  local wav_count
  wav_count="$(find "${staging_root}" -type f -iname '*.wav' | wc -l)"
  printf '%s\n' "${wav_count}" > "${RUN_DIR}/${split}_staging_wav_count.txt"
  if [[ "${wav_count}" -ne "${expected_wav}" ]]; then
    echo "[ERROR] ${split} extracted WAV count mismatch: ${wav_count} != ${expected_wav}" >&2
    exit 10
  fi
  mv "${staging_root}" "${final_root}"
  {
    printf 'archive_md5=%s\n' "${expected_md5}"
    printf 'expected_wav_files=%s\n' "${expected_wav}"
    printf 'git_commit=%s\n' "$(cat "${RUN_DIR}/git_commit.txt")"
    printf 'completed_at=%s\n' "$(date -Is)"
  } > "${marker}"
}

extract_split development "${DEVELOPMENT_ARCHIVE}" "${EXPECTED_MD5[0]}" 3839
extract_split validation "${VALIDATION_ARCHIVE}" "${EXPECTED_MD5[1]}" 1045

set +e
python scripts/validate_clotho_trainval.py \
  --development-root "${EXTRACT_ROOT}/development" \
  --validation-root "${EXTRACT_ROOT}/validation" \
  --development-captions "${DEVELOPMENT_CAPTIONS}" \
  --validation-captions "${VALIDATION_CAPTIONS}" \
  --development-metadata "${DEVELOPMENT_METADATA}" \
  --validation-metadata "${VALIDATION_METADATA}" \
  --development-manifest-output "${MANIFEST_ROOT}/clotho_development_manifest.jsonl" \
  --validation-manifest-output "${MANIFEST_ROOT}/clotho_validation_manifest.jsonl" \
  --statistics-output "${RUN_DIR}/data_statistics.json" \
  --expected-development 3839 \
  --expected-validation 1045
VALIDATION_EXIT_CODE=$?
set -e
printf '%s\n' "${VALIDATION_EXIT_CODE}" > "${RUN_DIR}/validation_exit_code.txt"
if [[ "${VALIDATION_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-10 validation failed; extraction and audit evidence were preserved." >&2
  exit "${VALIDATION_EXIT_CODE}"
fi

find "${EXTRACT_ROOT}/development" -type f -iname '*.wav' -printf '%s %p\n' \
  | sort > "${RUN_DIR}/development_audio_files.txt"
find "${EXTRACT_ROOT}/validation" -type f -iname '*.wav' -printf '%s %p\n' \
  | sort > "${RUN_DIR}/validation_audio_files.txt"
sha256sum \
  "${MANIFEST_ROOT}/clotho_development_manifest.jsonl" \
  "${MANIFEST_ROOT}/clotho_validation_manifest.jsonl" \
  > "${RUN_DIR}/manifest_sha256.txt"
df -hT "${DATASET_ROOT}" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${DATASET_ROOT}" > "${RUN_DIR}/dataset_size.txt"

echo "[INFO] DATA-10 Clotho development/validation validation completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
echo "[INFO] Canonical manifests: ${MANIFEST_ROOT}/clotho_{development,validation}_manifest.jsonl"
