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
FIQA_ROOT="${DATA_ROOT}/fiqa_mteb"
RESOURCE_MANIFEST="${ROOT_DIR}/configs/resources/data12_squtr.json"
STRUCTURE_MANIFEST="${ROOT_DIR}/configs/resources/data13_squtr_structure.json"
EXPERIMENT_CONFIG="${ROOT_DIR}/configs/asr_uncertainty_reranking/main_experiment.json"
MANIFEST_OUTPUT="${MANIFEST_ROOT}/squtr_en_fiqa_nq_audio_query_manifest.jsonl"
PROBE_CACHE="${MANIFEST_ROOT}/.squtr_audio_probe_cache_en_fiqa_nq_v1.jsonl"
LEGACY_LOCK_FILE="${MANIFEST_ROOT}/.data13c_content_recovery.lock"
LOCK_FILE="${MANIFEST_ROOT}/.data13d_en_fiqa_nq.lock"
RUN_ID="data13d_squtr_fiqa_nq_validation_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[ERROR] Refusing DATA-13D from a dirty Git worktree." >&2
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
echo "[INFO] Selected subsets: en/fiqa, en/nq"
echo "[INFO] Manifest: ${MANIFEST_OUTPUT}"
echo "[INFO] Probe cache: ${PROBE_CACHE}"
echo "[INFO] GPU disabled"
echo "[INFO] No archive member will be extracted, deleted, or overwritten"
echo "[INFO] Legacy DATA-13C lock is evidence only and will not be modified"

if [[ -e "${LEGACY_LOCK_FILE}" ]]; then
  stat "${LEGACY_LOCK_FILE}" > "${RUN_DIR}/legacy_lock_stat.txt" 2>&1 || true
else
  echo "[ABSENT] ${LEGACY_LOCK_FILE}" > "${RUN_DIR}/legacy_lock_stat.txt"
fi

exec 9>> "${LOCK_FILE}"
if ! flock -n 9; then
  echo "[ERROR] Another target-subset DATA-13D run owns ${LOCK_FILE}" >&2
  if [[ -s "${LOCK_FILE}" ]]; then
    sed 's/^/[ERROR]   /' "${LOCK_FILE}" >&2
  fi
  exit 20
fi
{
  printf 'schema=oea_data13d_target_subset_lock_v1\n'
  printf 'scope=en/fiqa,en/nq\n'
  printf 'last_acquired_at=%s\n' "$(date -Is)"
  printf 'last_owner_host=%s\n' "$(hostname)"
  printf 'last_owner_pid=%s\n' "$$"
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
  echo "[ERROR] DATA-13D refused extraction reuse." >&2
  exit "${REUSE_EXIT_CODE}"
fi

set +e
python scripts/validate_squtr_extracted.py \
  --extract-root "${EXTRACT_ROOT}" \
  --structure-manifest "${STRUCTURE_MANIFEST}" \
  --subset en/fiqa \
  --subset en/nq \
  --manifest-output "${MANIFEST_OUTPUT}" \
  --statistics-output "${RUN_DIR}/data_statistics.json" \
  --probe-cache "${PROBE_CACHE}" \
  --probe-cache-git-commit "${GIT_COMMIT}"
VALIDATION_EXIT_CODE=$?
set -e
printf '%s\n' "${VALIDATION_EXIT_CODE}" > "${RUN_DIR}/validation_exit_code.txt"
if [[ "${VALIDATION_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-13D target-subset content validation failed." >&2
  exit "${VALIDATION_EXIT_CODE}"
fi

set +e
python scripts/audit_asrur_fiqa_data.py \
  --config "${EXPERIMENT_CONFIG}" \
  --fiqa-root "${FIQA_ROOT}" \
  --squtr-manifest "${MANIFEST_OUTPUT}" \
  --output-dir "${RUN_DIR}/fiqa_consistency" \
  --require-audio-files
FIQA_AUDIT_EXIT_CODE=$?
set -e
printf '%s\n' "${FIQA_AUDIT_EXIT_CODE}" > "${RUN_DIR}/fiqa_audit_exit_code.txt"

df -hT "${DATASET_ROOT}" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${EXTRACT_ROOT}" "${PROBE_CACHE}" "${MANIFEST_OUTPUT}" \
  > "${RUN_DIR}/artifact_sizes.txt" 2>&1 || true
sha256sum "${PROBE_CACHE}" > "${RUN_DIR}/probe_cache_sha256.txt"
sha256sum "${MANIFEST_OUTPUT}" > "${RUN_DIR}/manifest_sha256.txt"

echo "===== DATA STATISTICS ====="
cat "${RUN_DIR}/data_statistics.json"
echo "===== FIQA CONSISTENCY AUDIT ====="
cat "${RUN_DIR}/fiqa_consistency/fiqa_data_audit.json"
echo "===== ARTIFACT HASHES ====="
cat "${RUN_DIR}/probe_cache_sha256.txt"
cat "${RUN_DIR}/manifest_sha256.txt"

if [[ "${FIQA_AUDIT_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] FiQA/SQuTR identity audit failed." >&2
  exit "${FIQA_AUDIT_EXIT_CODE}"
fi

echo "[INFO] DATA-13D FiQA/NQ validation and FiQA identity audit completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
