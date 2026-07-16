#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/model_cache/oea}"
RUN_ID="model02_download_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
MANIFEST="${ROOT_DIR}/configs/resources/model02_qwen3b_ac.json"

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
df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available." | tee "${RUN_DIR}/stderr.log" >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[ERROR] Conda environment '${ENV_NAME}' does not exist." | tee "${RUN_DIR}/stderr.log" >&2
  exit 3
fi

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] Resource manifest: ${MANIFEST}"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

set +e
python scripts/download_model_assets.py \
  --manifest "${MANIFEST}" \
  --model-root "${MODEL_ROOT}" \
  --output "${RUN_DIR}/download_manifest.json"
DOWNLOAD_EXIT_CODE=$?
set -e
printf '%s\n' "${DOWNLOAD_EXIT_CODE}" > "${RUN_DIR}/download_exit_code.txt"

df -hT "$(dirname "${MODEL_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${MODEL_ROOT}" > "${RUN_DIR}/model_root_size.txt" 2>&1 || true

if [[ "${DOWNLOAD_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] MODEL-02 download failed; Hugging Face partial files were preserved for resume." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${DOWNLOAD_EXIT_CODE}"
fi

echo "[INFO] MODEL-02 download and SHA256 verification completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
