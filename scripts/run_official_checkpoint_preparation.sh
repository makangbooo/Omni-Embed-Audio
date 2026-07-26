#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <variant_id> [--inspect-only|--verify-existing-derived]" >&2
  exit 2
fi

VARIANT_ID=$1
MODE=${2:-}
if [[ ! "${VARIANT_ID}" =~ ^[a-z0-9]+(_[a-z0-9]+)*$ ]]; then
  echo "[ERROR] Invalid variant ID: ${VARIANT_ID}" >&2
  exit 2
fi
if [[ -n "${MODE}" && "${MODE}" != "--inspect-only" && "${MODE}" != "--verify-existing-derived" ]]; then
  echo "[ERROR] Optional mode must be --inspect-only or --verify-existing-derived." >&2
  exit 2
fi

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
REGISTRY="${ROOT_DIR}/configs/checkpoints/official_oea_checkpoints.json"
RUN_ID="official_checkpoint_preparation_${VARIANT_ID}_$(date +%Y%m%d_%H%M%S)"
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
free -h > "${RUN_DIR}/memory_before.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Variant: ${VARIANT_ID}"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] GPU disabled"
echo "[INFO] Mode: ${MODE:---inspect-and-extract}"

ARGUMENTS=(
  --registry "${REGISTRY}"
  --variant "${VARIANT_ID}"
  --model-root "${MODEL_ROOT}"
  --output-dir "${RUN_DIR}"
)
if [[ "${MODE}" == "--inspect-only" ]]; then
  ARGUMENTS+=(--inspect-only)
elif [[ "${MODE}" == "--verify-existing-derived" ]]; then
  ARGUMENTS+=(--verify-existing-derived)
fi

set +e
python scripts/prepare_official_oea_checkpoint.py "${ARGUMENTS[@]}"
PREPARATION_EXIT_CODE=$?
set -e
printf '%s\n' "${PREPARATION_EXIT_CODE}" > "${RUN_DIR}/preparation_exit_code.txt"

free -h > "${RUN_DIR}/memory_after.txt"
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true

if [[ "${PREPARATION_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] Official checkpoint preparation failed; all audit files were retained." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${PREPARATION_EXIT_CODE}"
fi

echo "[INFO] Official checkpoint preparation completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
