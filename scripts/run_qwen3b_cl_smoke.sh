#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="oea-repro"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/model_cache/oea}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/eval/qwen3b_cl_smoke.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/experiments}"
SEED=42
RUN_ID="oea_qwen3b_cl_clotho5_smoke_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"

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
  printf 'MODEL_ROOT=%q CONFIG=%q OUTPUT_ROOT=%q SMOKE_CUDA_DEVICE=%q ' \
    "${MODEL_ROOT}" "${CONFIG}" "${OUTPUT_ROOT}" "${SMOKE_CUDA_DEVICE:-0}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
nvidia-smi > "${RUN_DIR}/gpu_info.txt"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

{
  date -Is
  uname -a
  printf 'CONDA_DEFAULT_ENV=%s\n' "${CONDA_DEFAULT_ENV:-}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${SMOKE_CUDA_DEVICE:-0}"
  python --version
  python - <<'PY'
import json
import platform
import torch
import transformers
import peft

print(json.dumps({
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_compiled_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
}, indent=2))
PY
  python -m pip freeze
} > "${RUN_DIR}/environment.txt"

export CUDA_VISIBLE_DEVICES="${SMOKE_CUDA_DEVICE:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Strict offline mode: enabled"
echo "[INFO] Model root: ${MODEL_ROOT}"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

set +e
python scripts/smoke_oea_qwen3b_cl.py \
  --config "${CONFIG}" \
  --model-root "${MODEL_ROOT}" \
  --output-dir "${RUN_DIR}"
SMOKE_EXIT_CODE=$?
set -e
printf '%s\n' "${SMOKE_EXIT_CODE}" > "${RUN_DIR}/smoke_exit_code.txt"
if [[ "${SMOKE_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] Qwen3B-Cl smoke exited with code ${SMOKE_EXIT_CODE}" >&2
  exit "${SMOKE_EXIT_CODE}"
fi

nvidia-smi > "${RUN_DIR}/gpu_info_after.txt"
echo "[INFO] Qwen3B-Cl offline smoke completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
