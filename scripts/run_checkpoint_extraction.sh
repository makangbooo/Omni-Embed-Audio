#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/model_cache/oea}"
SOURCE="${SOURCE:-${MODEL_ROOT}/OEA-Qwen3B-Cl/step_40.pt}"
DESTINATION="${DESTINATION:-${MODEL_ROOT}/OEA-Qwen3B-Cl/step_40_inference_only.pt}"
RUN_ID="checkpoint_extraction_$(date +%Y%m%d_%H%M%S)"
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
  printf 'MODEL_ROOT=%q SOURCE=%q DESTINATION=%q ' \
    "${MODEL_ROOT}" "${SOURCE}" "${DESTINATION}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
free -h > "${RUN_DIR}/memory_before.txt"
for FILE in memory.current memory.max memory.events; do
  if [[ -r "/sys/fs/cgroup/${FILE}" ]]; then
    {
      echo "--- ${FILE} ---"
      cat "/sys/fs/cgroup/${FILE}"
    } >> "${RUN_DIR}/cgroup_memory_before.txt"
  fi
done

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Source checkpoint: ${SOURCE}"
echo "[INFO] Destination checkpoint: ${DESTINATION}"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

set +e
python scripts/extract_oea_inference_checkpoint.py \
  --source "${SOURCE}" \
  --destination "${DESTINATION}" \
  --audit-output "${RUN_DIR}/extraction_manifest.json" \
  --expected-source-size 9466833858 \
  --expected-source-sha256 d5f2648c19b07fe5b33c22873098f9b9d749dfe2fb48e5d3dedfcea827c39b5c \
  --expected-lora-tensors 544 \
  --expected-lora-bytes 50462720
EXTRACTION_EXIT_CODE=$?
set -e
printf '%s\n' "${EXTRACTION_EXIT_CODE}" > "${RUN_DIR}/extraction_exit_code.txt"
if [[ "${EXTRACTION_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] Checkpoint extraction exited with code ${EXTRACTION_EXIT_CODE}" >&2
  exit "${EXTRACTION_EXIT_CODE}"
fi

free -h > "${RUN_DIR}/memory_after.txt"
ls -lh "${SOURCE}" "${DESTINATION}" > "${RUN_DIR}/checkpoint_sizes.txt"
echo "[INFO] Checkpoint extraction completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
