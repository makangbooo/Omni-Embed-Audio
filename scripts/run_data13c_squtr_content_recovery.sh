#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
DATASET_ROOT="${DATA_ROOT}/squtr"
EXTRACT_ROOT="${DATASET_ROOT}/extracted"
MANIFEST_ROOT="${DATASET_ROOT}/manifests"
RESOURCE_MANIFEST="${ROOT_DIR}/configs/resources/data12_squtr.json"
STRUCTURE_MANIFEST="${ROOT_DIR}/configs/resources/data13_squtr_structure.json"
MANIFEST_OUTPUT="${MANIFEST_ROOT}/squtr_audio_query_manifest.jsonl"
PROBE_CACHE="${MANIFEST_ROOT}/.squtr_audio_probe_cache_v2.jsonl"
RUN_ID="data13c_squtr_content_recovery_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOCK_FILE="${MANIFEST_ROOT}/.data13c_content_recovery.lock"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[ERROR] Refusing DATA-13C from a dirty Git worktree." >&2
  git status --short --untracked-files=all >&2
  exit 2
fi

mkdir -p "${RUN_DIR}" "${MANIFEST_ROOT}"
record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
}
trap record_wrapper_exit EXIT

{
  printf 'DATA_ROOT=%q ' "${DATA_ROOT}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
GIT_COMMIT="$(git rev-parse HEAD)"
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
echo "[INFO] Extract root: ${EXTRACT_ROOT}"
echo "[INFO] Manifest: ${MANIFEST_OUTPUT}"
echo "[INFO] Probe cache: ${PROBE_CACHE}"
echo "[INFO] GPU disabled"
echo "[INFO] This recovery never deletes attempt-1 evidence or extracted data"

exec 9>> "${LOCK_FILE}"
if ! flock -n 9; then
  echo "[ERROR] Another DATA-13C recovery owns ${LOCK_FILE}" >&2
  if [[ -s "${LOCK_FILE}" ]]; then
    echo "[ERROR] Last recorded lock owner metadata:" >&2
    sed 's/^/[ERROR]   /' "${LOCK_FILE}" >&2
  else
    echo "[ERROR] Lock owner metadata is unavailable (legacy empty lock file)." >&2
  fi
  exit 20
fi
{
  printf 'schema=oea_data13c_lock_v1\n'
  printf 'last_acquired_at=%s\n' "$(date -Is)"
  printf 'last_owner_host=%s\n' "$(hostname)"
  printf 'last_owner_pid=%s\n' "$$"
  printf 'last_owner_ppid=%s\n' "${PPID}"
  printf 'last_owner_git_commit=%s\n' "${GIT_COMMIT}"
  printf 'last_owner_run_dir=%s\n' "${RUN_DIR}"
} > "${LOCK_FILE}"

set +e
python scripts/verify_squtr_extraction_completion.py \
  --extract-root "${EXTRACT_ROOT}" \
  --resource-manifest "${RESOURCE_MANIFEST}" \
  --structure-manifest "${STRUCTURE_MANIFEST}" \
  --output "${RUN_DIR}/extraction_reuse_report.json"
REUSE_EXIT_CODE=$?
set -e
printf '%s\n' "${REUSE_EXIT_CODE}" > "${RUN_DIR}/extraction_reuse_exit_code.txt"
if [[ "${REUSE_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-13C refused extraction reuse." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${REUSE_EXIT_CODE}"
fi

set +e
python scripts/validate_squtr_extracted.py \
  --extract-root "${EXTRACT_ROOT}" \
  --structure-manifest "${STRUCTURE_MANIFEST}" \
  --manifest-output "${MANIFEST_OUTPUT}" \
  --statistics-output "${RUN_DIR}/data_statistics.json" \
  --probe-cache "${PROBE_CACHE}" \
  --probe-cache-git-commit "${GIT_COMMIT}"
VALIDATION_EXIT_CODE=$?
set -e
printf '%s\n' "${VALIDATION_EXIT_CODE}" > "${RUN_DIR}/validation_exit_code.txt"

df -hT "${DATASET_ROOT}" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${EXTRACT_ROOT}" "${PROBE_CACHE}" "${MANIFEST_OUTPUT}" \
  > "${RUN_DIR}/artifact_sizes.txt" 2>&1 || true
if [[ -f "${PROBE_CACHE}" ]]; then
  sha256sum "${PROBE_CACHE}" > "${RUN_DIR}/probe_cache_sha256.txt"
fi
if [[ -f "${MANIFEST_OUTPUT}" ]]; then
  sha256sum "${MANIFEST_OUTPUT}" > "${RUN_DIR}/manifest_sha256.txt"
fi

echo "===== EXTRACTION REUSE REPORT ====="
cat "${RUN_DIR}/extraction_reuse_report.json"
echo "===== DATA STATISTICS ====="
cat "${RUN_DIR}/data_statistics.json"
echo "===== ARTIFACT SIZES ====="
cat "${RUN_DIR}/artifact_sizes.txt"
if [[ -f "${RUN_DIR}/probe_cache_sha256.txt" ]]; then
  echo "===== PROBE CACHE SHA256 ====="
  cat "${RUN_DIR}/probe_cache_sha256.txt"
fi
if [[ -f "${RUN_DIR}/manifest_sha256.txt" ]]; then
  echo "===== MANIFEST SHA256 ====="
  cat "${RUN_DIR}/manifest_sha256.txt"
fi

if [[ "${VALIDATION_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-13C content validation failed; resumable probe cache preserved." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${VALIDATION_EXIT_CODE}"
fi

echo "[INFO] DATA-13C extraction reuse and content validation completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
