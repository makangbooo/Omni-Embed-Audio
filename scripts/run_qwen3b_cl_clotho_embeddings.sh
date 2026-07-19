#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/model_cache/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MANIFEST="${DATA_ROOT}/clotho_v2.1/manifests/clotho_evaluation_manifest.jsonl"
PROTOCOL_CONFIG="${ROOT_DIR}/configs/eval/qwen3b_cl_clotho_embeddings.json"
MODEL_LOCK="${MODEL_LOCK:-${ROOT_DIR}/results/model_locks/oea_qwen3b_cl.json}"
RUN_ID="${RUN_ID:-oea_qwen3b_clotho_embeddings_seed42_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${RESULT_ROOT:-${ROOT_DIR}/results/raw}/${RUN_ID}"
RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_embedding_config.json"
ATTEMPT_ID="attempt_$(date +%Y%m%d_%H%M%S)"
ATTEMPT_DIR="${OUTPUT_DIR}/attempts/${ATTEMPT_ID}"

if [[ -e "${ATTEMPT_DIR}" ]]; then
  echo "[ERROR] Attempt directory already exists; refusing to overwrite: ${ATTEMPT_DIR}" >&2
  exit 2
fi
mkdir -p "${ATTEMPT_DIR}"
record_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${ATTEMPT_DIR}/exit_code.txt"
}
trap record_exit EXIT
exec > >(tee "${ATTEMPT_DIR}/stdout.log") \
  2> >(tee "${ATTEMPT_DIR}/stderr.log" >&2)

{
  printf 'MODEL_ROOT=%q DATA_ROOT=%q MODEL_LOCK=%q RUN_ID=%q ' \
    "${MODEL_ROOT}" "${DATA_ROOT}" "${MODEL_LOCK}" "${RUN_ID}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${ATTEMPT_DIR}/command.sh"
git rev-parse HEAD > "${ATTEMPT_DIR}/git_commit.txt"
git status --short > "${ATTEMPT_DIR}/git_status.txt"
if [[ -s "${ATTEMPT_DIR}/git_status.txt" ]]; then
  echo "[ERROR] Formal embedding generation requires a clean Git worktree." >&2
  exit 3
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

{
  printf 'timestamp=%s\n' "$(date -Is)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'conda_prefix=%s\n' "${CONDA_PREFIX:-}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
  python --version
  python -c 'import numpy, torch, transformers, peft; print(f"numpy={numpy.__version__}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"peft={peft.__version__}")'
} > "${ATTEMPT_DIR}/environment.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
  --format=csv > "${ATTEMPT_DIR}/gpu_info.txt"
python scripts/validate_single_bf16_gpu.py \
  --output "${ATTEMPT_DIR}/gpu_preflight.json"

echo "[INFO] Run ID: ${RUN_ID}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"
echo "[INFO] Attempt directory: ${ATTEMPT_DIR}"
echo "[INFO] Resume by exporting the same RUN_ID and running this wrapper again."

python scripts/build_official_oea_eval_config.py \
  --protocol-config "${PROTOCOL_CONFIG}" \
  --model-lock "${MODEL_LOCK}" \
  --output "${RESOLVED_CONFIG}" \
  > "${ATTEMPT_DIR}/config_resolution.json"

python scripts/generate_oea_embeddings.py \
  --config "${RESOLVED_CONFIG}" \
  --model-root "${MODEL_ROOT}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --attempt-dir "${ATTEMPT_DIR}"
