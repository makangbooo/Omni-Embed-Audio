#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <vanilla_backbone_id> --smoke" >&2
  echo "       $0 <vanilla_backbone_id> --full <smoke_metrics.json>" >&2
  exit 2
fi

BACKBONE_ID=$1
RUN_MODE=$2
case "${BACKBONE_ID}" in
  vanilla_nemotron_3b)
    CONFIG_STEM="vanilla_nemotron_3b"
    ;;
  vanilla_qwen2_5_omni_3b)
    CONFIG_STEM="vanilla_qwen2_5_omni_3b"
    ;;
  vanilla_qwen2_5_omni_7b)
    CONFIG_STEM="vanilla_qwen2_5_omni_7b"
    ;;
  *)
    echo "[ERROR] Unknown vanilla backbone: ${BACKBONE_ID}" >&2
    exit 2
    ;;
esac

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/model_cache/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MODEL_LOCK="${MODEL_LOCK:-${ROOT_DIR}/results/model_locks/${BACKBONE_ID}.json}"
case "${RUN_MODE}" in
  --smoke)
    if [[ $# -ne 2 ]]; then
      echo "[ERROR] --smoke does not accept a smoke-metrics argument." >&2
      exit 2
    fi
    PROTOCOL_CONFIG="${ROOT_DIR}/configs/eval/${CONFIG_STEM}_clotho_smoke_embeddings.json"
    MANIFEST="${ROOT_DIR}/configs/eval/fixtures/vanilla_clotho_5_manifest.jsonl"
    DEFAULT_RUN_PREFIX="${BACKBONE_ID}_clotho_smoke_seed42"
    ;;
  --full)
    if [[ $# -ne 3 ]]; then
      echo "[ERROR] --full requires an explicit completed smoke metrics path." >&2
      exit 2
    fi
    SMOKE_METRICS=$3
    PROTOCOL_CONFIG="${ROOT_DIR}/configs/eval/${CONFIG_STEM}_clotho_embeddings.json"
    MANIFEST="${DATA_ROOT}/clotho_v2.1/manifests/clotho_evaluation_manifest.jsonl"
    DEFAULT_RUN_PREFIX="${BACKBONE_ID}_clotho_embeddings_seed42"
    ;;
  *)
    echo "[ERROR] Run mode must be exactly --smoke or --full." >&2
    exit 2
    ;;
esac
RUN_ID="${RUN_ID:-${DEFAULT_RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
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
  echo "[ERROR] Formal vanilla embedding generation requires a clean Git worktree." >&2
  cat "${ATTEMPT_DIR}/git_status.txt" >&2
  exit 3
fi
if [[ ! -f "${MODEL_LOCK}" ]]; then
  echo "[ERROR] Committed vanilla model lock is missing: ${MODEL_LOCK}" >&2
  exit 4
fi
if ! git ls-files --error-unmatch -- "${MODEL_LOCK#${ROOT_DIR}/}" >/dev/null 2>&1; then
  echo "[ERROR] Vanilla model lock must be committed before GPU generation: ${MODEL_LOCK}" >&2
  exit 4
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
  python -c 'import numpy, torch, transformers; print(f"numpy={numpy.__version__}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}")'
} > "${ATTEMPT_DIR}/environment.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
  --format=csv > "${ATTEMPT_DIR}/gpu_info.txt"

echo "[INFO] Backbone: ${BACKBONE_ID}"
echo "[INFO] Run mode: ${RUN_MODE}"
echo "[INFO] Run ID: ${RUN_ID}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"
echo "[INFO] Attempt directory: ${ATTEMPT_DIR}"
echo "[INFO] Base-only path: no OEA checkpoint, LoRA, or projection head"
echo "[INFO] Resume by exporting the same RUN_ID and running this wrapper again."

if [[ "${RUN_MODE}" == "--full" ]]; then
  python scripts/verify_vanilla_smoke_gate.py \
    --metrics "${SMOKE_METRICS}" \
    --backbone "${BACKBONE_ID}" \
    --model-lock "${MODEL_LOCK}" \
    --output "${ATTEMPT_DIR}/smoke_gate.json"
fi

python scripts/build_vanilla_backbone_eval_config.py \
  --protocol-config "${PROTOCOL_CONFIG}" \
  --model-lock "${MODEL_LOCK}" \
  --output "${RESOLVED_CONFIG}" \
  > "${ATTEMPT_DIR}/config_resolution.json"

python scripts/generate_vanilla_backbone_embeddings.py \
  --config "${RESOLVED_CONFIG}" \
  --model-root "${MODEL_ROOT}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --attempt-dir "${ATTEMPT_DIR}"
