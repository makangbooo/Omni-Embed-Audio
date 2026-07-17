#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <variant_id> [--verify-existing-derived]" >&2
  exit 2
fi

VARIANT_ID=$1
MODE=${2:-}
if [[ ! "${VARIANT_ID}" =~ ^[a-z0-9]+(_[a-z0-9]+)*$ ]]; then
  echo "[ERROR] Invalid variant ID: ${VARIANT_ID}" >&2
  exit 2
fi
if [[ -n "${MODE}" && "${MODE}" != "--verify-existing-derived" ]]; then
  echo "[ERROR] Optional mode must be --verify-existing-derived." >&2
  exit 2
fi

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/model_cache/oea}"
REGISTRY="${ROOT_DIR}/configs/checkpoints/official_oea_checkpoints.json"
MODEL_LOCK="${ROOT_DIR}/results/model_locks/${VARIANT_ID}.json"
RUN_ID="official_model_pipeline_${VARIANT_ID}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
PIPELINE_DIR="${RUN_DIR}/pipeline"

if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse pipeline run directory: ${RUN_DIR}" >&2
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
  echo "[ERROR] Official model pipeline requires a clean Git worktree." >&2
  cat "${RUN_DIR}/git_status.txt" >&2
  exit 2
fi
if [[ -e "${MODEL_LOCK}" ]]; then
  echo "[ERROR] Refusing to overwrite existing model lock: ${MODEL_LOCK}" >&2
  exit 2
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Variant: ${VARIANT_ID}"
echo "[INFO] Mode: ${MODE:---extract-new-derived}"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] Model lock: ${MODEL_LOCK}"
echo "[INFO] GPU disabled"
echo "[INFO] Stages: exact resource audit -> checkpoint preparation -> model lock"

ARGUMENTS=(
  --registry "${REGISTRY}"
  --variant "${VARIANT_ID}"
  --model-root "${MODEL_ROOT}"
  --output-dir "${PIPELINE_DIR}"
  --lock-output "${MODEL_LOCK}"
)
if [[ "${MODE}" == "--verify-existing-derived" ]]; then
  ARGUMENTS+=(--verify-existing-derived)
fi

set +e
python scripts/run_official_oea_model_pipeline.py "${ARGUMENTS[@]}"
PIPELINE_EXIT_CODE=$?
set -e
printf '%s\n' "${PIPELINE_EXIT_CODE}" > "${RUN_DIR}/pipeline_exit_code.txt"

free -h > "${RUN_DIR}/memory_after.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true

python -m json.tool "${PIPELINE_DIR}/pipeline_manifest.json" || true

if [[ "${PIPELINE_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] Official model pipeline failed; all audit evidence was retained." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${PIPELINE_EXIT_CODE}"
fi

echo "[INFO] Official model pipeline completed"
echo "[INFO] Review and commit the small model lock: ${MODEL_LOCK}"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
