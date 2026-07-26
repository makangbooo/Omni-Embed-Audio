#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="${ASRUR_ENV_NAME:-oea-repro}"
MODELS_ROOT="${MODELS_ROOT:-/home/jg525/models}"
MANIFEST="${ROOT_DIR}/configs/asr_uncertainty_reranking/resources/models.json"
RUN_ID="${RUN_ID:-asrur_d2_d4_model_audit_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
}
trap record_wrapper_exit EXIT

{
  printf 'MODELS_ROOT=%q RUN_ID=%q ' "${MODELS_ROOT}" "${RUN_ID}"
  printf '%q\n' "$0" "$@"
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short --untracked-files=all > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
df -hT "${MODELS_ROOT}" > "${RUN_DIR}/disk.txt" 2>&1 || true

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

if [[ -s "${RUN_DIR}/git_status.txt" ]]; then
  echo "[ERROR] Model audit requires a clean Git worktree." >&2
  cat "${RUN_DIR}/git_status.txt" >&2
  exit 2
fi
if [[ ! -d "${MODELS_ROOT}" ]]; then
  echo "[ERROR] Missing model root: ${MODELS_ROOT}" >&2
  exit 3
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] Missing resource manifest: ${MANIFEST}" >&2
  exit 3
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Model root: ${MODELS_ROOT}"
echo "[INFO] Manifest: ${MANIFEST}"
echo "[INFO] GPU disabled"
echo "[INFO] Network access is not used"
echo "[INFO] Git revision, file size, Git-LFS pointer, and SHA256 are strict"

set +e
python scripts/verify_asrur_model_assets.py \
  --manifest "${MANIFEST}" \
  --model-root "${MODELS_ROOT}" \
  --output "${RUN_DIR}/model_audit.json"
AUDIT_EXIT_CODE=$?
set -e
printf '%s\n' "${AUDIT_EXIT_CODE}" > "${RUN_DIR}/audit_exit_code.txt"

if [[ "${AUDIT_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] One or more D2-D4 assets failed strict verification." >&2
  echo "[ERROR] No model file was modified or deleted." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${AUDIT_EXIT_CODE}"
fi

echo "[INFO] D2-D4 strict offline verification completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
