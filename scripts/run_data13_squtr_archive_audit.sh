#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
ARCHIVE="${DATA_ROOT}/squtr/source/source_data.zip"
RESOURCE_MANIFEST="${ROOT_DIR}/configs/resources/data12_squtr.json"
STRUCTURE_MANIFEST="${ROOT_DIR}/configs/resources/data13_squtr_structure.json"
RUN_ID="data13_squtr_archive_audit_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

mkdir -p "${RUN_DIR}"
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
echo "[INFO] Archive: ${ARCHIVE}"
echo "[INFO] GPU disabled"
echo "[INFO] DATA-13A is read-only and performs no extraction"

set +e
python scripts/audit_squtr_archive.py \
  --archive "${ARCHIVE}" \
  --resource-manifest "${RESOURCE_MANIFEST}" \
  --structure-manifest "${STRUCTURE_MANIFEST}" \
  --output "${RUN_DIR}/archive_audit.json"
AUDIT_EXIT_CODE=$?
set -e
printf '%s\n' "${AUDIT_EXIT_CODE}" > "${RUN_DIR}/audit_exit_code.txt"

df -hT "$(dirname "${DATA_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
cat "${RUN_DIR}/archive_audit.json"

if [[ "${AUDIT_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-13A archive audit failed; no files were extracted." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${AUDIT_EXIT_CODE}"
fi

echo "[INFO] DATA-13A read-only archive audit completed"
echo "[INFO] No archive member was extracted"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
