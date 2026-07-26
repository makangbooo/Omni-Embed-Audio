#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
RUN_ID="model03_model04_audit_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

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
  printf 'MODEL_ROOT=%q ' "${MODEL_ROOT}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk.txt" 2>&1 || true

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] GPU disabled"
echo "[INFO] Read-only audit: no models will be downloaded, repaired, or deleted"
echo "[INFO] Persistent Hugging Face .lock files are ignored; .incomplete files are reported"

set +e
python scripts/audit_model_resources.py \
  --manifest configs/resources/model03_nemo3b.json \
  --manifest configs/resources/model04_qwen7b.json \
  --model-root "${MODEL_ROOT}" \
  --output "${RUN_DIR}/model_resource_audit.json"
AUDIT_EXIT_CODE=$?
set -e
printf '%s\n' "${AUDIT_EXIT_CODE}" > "${RUN_DIR}/audit_exit_code.txt"

python -m json.tool "${RUN_DIR}/model_resource_audit.json"

case "${AUDIT_EXIT_CODE}" in
  0)
    echo "[INFO] MODEL-03 and MODEL-04 are complete and match their pinned snapshots"
    ;;
  2)
    echo "[WARN] MODEL-03 or MODEL-04 is incomplete; no files were changed" >&2
    ;;
  *)
    echo "[ERROR] MODEL-03 or MODEL-04 failed provenance/content validation" >&2
    ;;
esac

echo "[INFO] Audit artifacts: ${RUN_DIR}"
exit "${AUDIT_EXIT_CODE}"
