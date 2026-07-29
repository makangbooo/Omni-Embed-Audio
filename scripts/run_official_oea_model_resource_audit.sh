#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <variant_id>" >&2
  exit 2
fi

VARIANT_ID=$1
if [[ ! "${VARIANT_ID}" =~ ^[a-z0-9]+(_[a-z0-9]+)*$ ]]; then
  echo "[ERROR] Invalid variant ID: ${VARIANT_ID}" >&2
  exit 2
fi

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
REGISTRY="${ROOT_DIR}/configs/checkpoints/official_oea_checkpoints.json"
RUN_ID="${RUN_ID:-official_model_resource_audit_${VARIANT_ID}_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "[ERROR] Invalid RUN_ID: ${RUN_ID}" >&2
  exit 2
fi
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse audit directory: ${RUN_DIR}" >&2
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
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
free -h > "${RUN_DIR}/memory_before.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

if [[ -s "${RUN_DIR}/git_status.txt" ]]; then
  echo "[ERROR] Official model audit requires a clean Git worktree." >&2
  cat "${RUN_DIR}/git_status.txt" >&2
  exit 2
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Variant: ${VARIANT_ID}"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] GPU disabled"
echo "[INFO] Read-only audit: no model file will be downloaded, repaired, or deleted"

set +e
python scripts/audit_official_oea_variant_resources.py \
  --registry "${REGISTRY}" \
  --variant "${VARIANT_ID}" \
  --model-root "${MODEL_ROOT}" \
  --output "${RUN_DIR}/model_resource_audit.json"
AUDIT_EXIT_CODE=$?
set -e
printf '%s\n' "${AUDIT_EXIT_CODE}" > "${RUN_DIR}/audit_exit_code.txt"

free -h > "${RUN_DIR}/memory_after.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true

python -m json.tool "${RUN_DIR}/model_resource_audit.json" || true

case "${AUDIT_EXIT_CODE}" in
  0)
    echo "[INFO] Selected base model and checkpoint match their pinned snapshots"
    ;;
  2)
    echo "[WARN] A selected resource is incomplete; no files were changed" >&2
    ;;
  *)
    echo "[ERROR] Selected resources failed provenance/content validation" >&2
    ;;
esac

echo "[INFO] Audit artifacts: ${RUN_DIR}"
exit "${AUDIT_EXIT_CODE}"
