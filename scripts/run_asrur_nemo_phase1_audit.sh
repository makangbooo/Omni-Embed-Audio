#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
VARIANT_ID="oea_nemo3b_cl"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
DERIVED_CHECKPOINT="${MODEL_ROOT}/OEA-Nemo3B-Cl/step_450_best_inference_only.pt"
RUN_ID="asrur_nemo_phase1_audit_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
PIPELINE_DIR="${RUN_DIR}/pipeline"
PORTABLE_LOCK="${RUN_DIR}/${VARIANT_ID}.portable_model_lock.json"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[ERROR] Refusing the Nemo Phase-1 audit from a dirty Git worktree." >&2
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
# The resource-audit stage must query the immutable revision metadata in order
# to reconstruct the complete expected file inventory. This is a small API
# request only: audit_official_oea_variant_resources.py has no download path.
unset HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_DISABLE_XET="1"

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)
echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Variant: ${VARIANT_ID}"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] Portable lock evidence: ${PORTABLE_LOCK}"
echo "[INFO] GPU disabled"
echo "[INFO] Network policy: fixed-revision Hugging Face metadata API only"
echo "[INFO] Model downloads remain disabled; all content verification is local"
echo "[INFO] No official source file will be modified or overwritten"

ARGUMENTS=(
  --variant "${VARIANT_ID}"
  --model-root "${MODEL_ROOT}"
  --output-dir "${PIPELINE_DIR}"
  --lock-output "${PORTABLE_LOCK}"
  --portable-lock-evidence
)
if [[ -f "${DERIVED_CHECKPOINT}" ]]; then
  echo "[INFO] Existing derived checkpoint found; strict verify-only mode selected"
  ARGUMENTS+=(--verify-existing-derived)
else
  echo "[INFO] Derived checkpoint absent; atomic extraction mode selected"
fi

set +e
python scripts/run_official_oea_model_pipeline.py "${ARGUMENTS[@]}"
PIPELINE_EXIT_CODE=$?
set -e
printf '%s\n' "${PIPELINE_EXIT_CODE}" > "${RUN_DIR}/pipeline_exit_code.txt"

free -h > "${RUN_DIR}/memory_after.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
if [[ -f "${PORTABLE_LOCK}" ]]; then
  sha256sum "${PORTABLE_LOCK}" > "${RUN_DIR}/portable_model_lock_sha256.txt"
  ls -lh "${PORTABLE_LOCK}" > "${RUN_DIR}/portable_model_lock_size.txt"
fi
if [[ -f "${DERIVED_CHECKPOINT}" ]]; then
  sha256sum "${DERIVED_CHECKPOINT}" > "${RUN_DIR}/derived_checkpoint_sha256.txt"
  ls -lh "${DERIVED_CHECKPOINT}" > "${RUN_DIR}/derived_checkpoint_size.txt"
fi

echo "===== PIPELINE MANIFEST ====="
python -m json.tool "${PIPELINE_DIR}/pipeline_manifest.json" || true
echo "===== MODEL RESOURCE AUDIT ====="
python -m json.tool "${PIPELINE_DIR}/model_resource_audit.json" || true
echo "===== PORTABLE MODEL LOCK ====="
python -m json.tool "${PORTABLE_LOCK}" || true
echo "===== HASHES ====="
cat "${RUN_DIR}/portable_model_lock_sha256.txt" 2>/dev/null || true
cat "${RUN_DIR}/derived_checkpoint_sha256.txt" 2>/dev/null || true

if [[ "${PIPELINE_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] Nemo Phase-1 audit failed; evidence was retained." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${PIPELINE_EXIT_CODE}"
fi

echo "[INFO] Nemo Phase-1 resource audit, checkpoint preparation, and portable lock completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
