#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
BACKBONE_ID="vanilla_nemotron_3b"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
RUN_ID="asrur_vanilla_nemo_phase2_audit_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
PIPELINE_DIR="${RUN_DIR}/pipeline"
PORTABLE_LOCK="${RUN_DIR}/${BACKBONE_ID}.portable_model_lock.json"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[ERROR] Refusing the vanilla-Nemotron audit from a dirty worktree." >&2
  git status --short --untracked-files=all >&2
  exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse run directory: ${RUN_DIR}" >&2
  exit 2
fi
mkdir "${RUN_DIR}"

record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
}
trap record_wrapper_exit EXIT

{
  printf 'MODEL_ROOT=%q ' "${MODEL_ROOT}"
  printf '%q\n' "$0" "$@"
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short --untracked-files=all > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
free -h > "${RUN_DIR}/memory_before.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""
# The auditor reads the immutable repository tree metadata but has no content
# download path. Clear all Hub offline aliases so that this small metadata
# request is not blocked by a parent shell.
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HUB_DISABLE_XET="1"

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)
echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Backbone: ${BACKBONE_ID}"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] Portable lock evidence: ${PORTABLE_LOCK}"
echo "[INFO] GPU disabled"
echo "[INFO] Network policy: fixed-revision Hugging Face metadata API only"
echo "[INFO] No model or dataset download is performed"
echo "[INFO] Existing model files are read and hashed but never modified"

set +e
python scripts/run_vanilla_backbone_model_pipeline.py \
  --backbone "${BACKBONE_ID}" \
  --model-root "${MODEL_ROOT}" \
  --output-dir "${PIPELINE_DIR}" \
  --lock-output "${PORTABLE_LOCK}" \
  --portable-lock-evidence
PIPELINE_EXIT_CODE=$?
set -e
printf '%s\n' "${PIPELINE_EXIT_CODE}" > "${RUN_DIR}/pipeline_exit_code.txt"

free -h > "${RUN_DIR}/memory_after.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
if [[ -f "${PORTABLE_LOCK}" ]]; then
  sha256sum "${PORTABLE_LOCK}" > "${RUN_DIR}/portable_model_lock_sha256.txt"
  ls -lh "${PORTABLE_LOCK}" > "${RUN_DIR}/portable_model_lock_size.txt"
fi

python -m json.tool "${PIPELINE_DIR}/pipeline_manifest.json" || true
python -m json.tool "${PIPELINE_DIR}/model_resource_audit.json" || true
python -m json.tool "${PORTABLE_LOCK}" || true
cat "${RUN_DIR}/portable_model_lock_sha256.txt" 2>/dev/null || true

if [[ "${PIPELINE_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] Vanilla-Nemotron audit failed; evidence was retained." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${PIPELINE_EXIT_CODE}"
fi

echo "[INFO] Vanilla-Nemotron resource audit and portable lock completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
