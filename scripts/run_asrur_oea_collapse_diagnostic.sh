#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || "$1" != "--execute" ]]; then
  echo "Usage: bash scripts/run_asrur_oea_collapse_diagnostic.sh --execute" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] Run inside a normal tmux session: tmux new -s asrur_oea_collapse_diag" >&2
  exit 3
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MODEL_ROOT="${OEA_MODEL_ROOT:-/home/jg525/models/oea}"
SQUTR_ROOT="${SQUTR_SOURCE_ROOT:-${DATA_ROOT}/squtr/extracted/source_data}"
SOURCE_MANIFEST="${SQUTR_MANIFEST:-${DATA_ROOT}/squtr/manifests/squtr_en_fiqa_nq_audio_query_manifest.jsonl}"
QUERY_COUNT="${ASRUR_OEA_DIAGNOSTIC_QUERY_COUNT:-256}"
NEGATIVE_COUNT="${ASRUR_OEA_DIAGNOSTIC_NEGATIVE_COUNT:-4096}"
BOOTSTRAP_ITERATIONS="${ASRUR_OEA_DIAGNOSTIC_BOOTSTRAP_ITERATIONS:-10000}"
SAMPLE_SEED="${ASRUR_OEA_DIAGNOSTIC_SAMPLE_SEED:-20260805}"
BOOTSTRAP_SEED="${ASRUR_OEA_DIAGNOSTIC_BOOTSTRAP_SEED:-20260805}"
RUN_ID="asrur_oea_collapse_matrix_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "===== ASRUR OEA FOUR-SPACE ATTRIBUTION MATRIX ====="
echo "variants=oea_nemo3b_cl,oea_qwen3b_cl subsets=fiqa,nq"
echo "query_count=${QUERY_COUNT} negative_document_count=${NEGATIVE_COUNT}"
echo "run_dir=${RUN_DIR}"
echo "training=disabled formal_cache_mutation=disabled checkpoint_selection=disabled network=disabled"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

DIRTY_FILES="$(git status --porcelain --untracked-files=all)"
UNRELATED_DIRTY="$(printf '%s\n' "${DIRTY_FILES}" | awk 'NF && $NF != "tests/test_benchmark_oea_query_encoder.py"')"
if [[ -n "${UNRELATED_DIRTY}" ]]; then
  echo "[ERROR] Formal diagnostic requires a clean Git worktree." >&2
  printf '%s\n' "${UNRELATED_DIRTY}" >&2
  exit 4
fi
python scripts/validate_single_bf16_gpu.py --output "${RUN_DIR}/gpu_preflight.json"

for variant in oea_nemo3b_cl oea_qwen3b_cl; do
  case "${variant}" in
    oea_nemo3b_cl) config="${NEMO_CONFIG:-${ROOT_DIR}/configs/eval/nemo3b_cl_clotho_embeddings.json}" ;;
    oea_qwen3b_cl) config="${QWEN_CONFIG:-${ROOT_DIR}/configs/eval/qwen3b_cl_clotho_embeddings.json}" ;;
  esac
  for subset in fiqa nq; do
    dataset_root="${DATA_ROOT}/${subset}_mteb"
    if [[ "${subset}" == "nq" ]]; then dataset_root="${SQUTR_ROOT}/en/nq"; fi
    output="${RUN_DIR}/${variant}_${subset}.json"
    echo "===== RUN ${variant} ${subset} ====="
    echo "dataset_root=${dataset_root}"
    python scripts/diagnose_asrur_oea_collapse.py \
      --resolved-model-config "${config}" \
      --model-root "${MODEL_ROOT}" \
      --dataset-root "${dataset_root}" \
      --subset "${subset}" \
      --audio-manifest "${SOURCE_MANIFEST}" \
      --query-count "${QUERY_COUNT}" \
      --negative-document-count "${NEGATIVE_COUNT}" \
      --sample-seed "${SAMPLE_SEED}" \
      --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
      --bootstrap-seed "${BOOTSTRAP_SEED}" \
      --output "${output}"
  done
done

echo "[INFO] Matrix complete: ${RUN_DIR}"
