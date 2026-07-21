#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/model_cache/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/raw}"
MANIFEST="${DATA_ROOT}/clotho_v2.1/manifests/clotho_evaluation_manifest.jsonl"
BENCHMARK_CONFIG="${ROOT_DIR}/configs/eval/qwen3b_cl_clotho_efficiency.json"
PROTOCOL_CONFIG="${ROOT_DIR}/configs/eval/qwen3b_cl_clotho_embeddings.json"
MODEL_LOCK="${ROOT_DIR}/results/model_locks/oea_qwen3b_cl.json"
RUN_ID="${RUN_ID:-oea_qwen3b_clotho_a100_efficiency_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${RESULT_ROOT}/${RUN_ID}"
RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_model_config.json"
ATTEMPT_DIR="${OUTPUT_DIR}/attempts/attempt_$(date +%Y%m%d_%H%M%S)"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] Output directory exists; benchmark runs are not resumed: ${OUTPUT_DIR}" >&2
  exit 3
fi
mkdir -p "${ATTEMPT_DIR}"
trap 'printf "%s\n" "$?" > "${ATTEMPT_DIR}/exit_code.txt"' EXIT

{
  printf 'MODEL_ROOT=%q DATA_ROOT=%q RESULT_ROOT=%q RUN_ID=%q ' \
    "${MODEL_ROOT}" "${DATA_ROOT}" "${RESULT_ROOT}" "${RUN_ID}"
  printf '%q\n' "$0" "$@"
} > "${ATTEMPT_DIR}/command.sh"
git rev-parse HEAD > "${ATTEMPT_DIR}/git_commit.txt"
git status --short > "${ATTEMPT_DIR}/git_status.txt"
if [[ -s "${ATTEMPT_DIR}/git_status.txt" ]]; then
  echo "[ERROR] Formal efficiency benchmark requires a clean Git worktree." >&2
  exit 4
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

{
  printf 'timestamp=%s\n' "$(date -Is)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'conda_prefix=%s\n' "${CONDA_PREFIX:-}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
  python --version
  python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}")'
  nvidia-smi
} > "${ATTEMPT_DIR}/environment.txt" 2>&1

exec > >(tee "${ATTEMPT_DIR}/stdout.log") \
  2> >(tee "${ATTEMPT_DIR}/stderr.log" >&2)

python scripts/build_official_oea_eval_config.py \
  --protocol-config "${PROTOCOL_CONFIG}" \
  --model-lock "${MODEL_LOCK}" \
  --output "${RESOLVED_CONFIG}" \
  > "${ATTEMPT_DIR}/config_resolution.json"

python scripts/benchmark_oea_query_encoder.py \
  --benchmark-config "${BENCHMARK_CONFIG}" \
  --model-config "${RESOLVED_CONFIG}" \
  --model-root "${MODEL_ROOT}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}"

echo "[INFO] Efficiency benchmark completed: ${OUTPUT_DIR}/metrics.json"
