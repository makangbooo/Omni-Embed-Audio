#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <clotho|audiocaps|mecat> <audio-embeddings.npz> <pairing-audit-root>" >&2
  exit 2
fi

DATASET_ID="$1"
AUDIO_NPZ="$(realpath "$2")"
PAIRING_ROOT="$(realpath "$3")"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
CHECKPOINT="${MODEL_ROOT}/m2d-clap/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"
CHECKPOINT_SHA256="238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1"
BERT_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bert-base-uncased"

case "${DATASET_ID}" in
  clotho)
    DATASET_LABEL="Clotho v2.1 evaluation"
    NEGATIVE_JSONL="${ROOT_DIR}/data/UIQ/clotho/clotho_evaluation_negative_queries.jsonl"
    EXPECTED_CANDIDATES=1045
    EXPECTED_QUERIES=542
    PAIRING_SHA256="c05101e76d6343a1446a0132bb16b041c03dcca7e9d96a62d61710177c9fb177"
    ;;
  audiocaps)
    DATASET_LABEL="AudioCaps v2 test"
    NEGATIVE_JSONL="${ROOT_DIR}/data/UIQ/audiocaps/audiocaps_test_negative_queries.jsonl"
    EXPECTED_CANDIDATES=975
    EXPECTED_QUERIES=630
    PAIRING_SHA256="8a773c5cd8519214baf1ebd715df89dd1b75ee685e563ba4bf9c0c8cf1e7effa"
    ;;
  mecat)
    DATASET_LABEL="MECAT-Caption 00A/test public 848"
    NEGATIVE_JSONL="${ROOT_DIR}/data/UIQ/mecat/mecat_negative_queries.jsonl"
    EXPECTED_CANDIDATES=848
    EXPECTED_QUERIES=409
    PAIRING_SHA256="e7c7311281681190f544c5d9d0eed348fd3378bbc4ce9c31358096da36bab8aa"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET_ID}" >&2
    exit 2
    ;;
esac

PAIRING_JSONL="${PAIRING_ROOT}/${DATASET_ID}/pairing_metadata.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="m2d_clap_${DATASET_ID}_negative_uiq_${STAMP}"
RESULT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}"
QUERY_DIR="${RESULT_DIR}/query_embeddings"
METRICS_DIR="${RESULT_DIR}/metrics"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"

mkdir -p "${QUERY_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_TIME="$(date -Is)"
START_EPOCH="$(date +%s)"
GIT_COMMIT="$(git rev-parse HEAD)"
CURRENT_STAGE="preflight"

format_duration() {
  local total="${1:-0}"
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

finish() {
  local rc="$1" status="$2" stage="$3" summary="$4" elapsed
  elapsed="$(($(date +%s) - START_EPOCH))"
  printf '%s\n' "${rc}" > "${RESULT_DIR}/exit_code.txt"
  echo "FINAL_RUN_RC=${rc}"
  echo "START_TIME=${START_TIME}"
  echo "END_TIME=$(date -Is)"
  echo "TOTAL_ELAPSED_SECONDS=${elapsed}"
  echo "TOTAL_ELAPSED=$(format_duration "${elapsed}")"
  echo "COMPLETION_STATUS=${status}"
  echo "FAILED_STAGE=${stage}"
  echo "ERROR_SUMMARY=${summary}"
  echo "RESULT_DIRECTORY=${RESULT_DIR}"
  echo "LOG_DIRECTORY=${LOG_DIR}"
  echo "METRICS_PATH=${METRICS_DIR}/metrics.json"
  return "${rc}"
}

fail() {
  local rc="$1" summary="$2"
  finish "${rc}" failed "${CURRENT_STAGE}" "${summary}"
  exit "${rc}"
}

run_stage() {
  local stage="$1" expected_seconds="$2"
  shift 2
  local stage_start pid now elapsed remaining rc
  stage_start="$(date +%s)"
  echo "STAGE_START=${stage}"
  "$@" &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 15
    now="$(date +%s)"
    elapsed="$((now - stage_start))"
    remaining="$((expected_seconds - elapsed))"
    if (( remaining < 0 )); then remaining=0; fi
    echo "PROGRESS stage=${stage} stage_elapsed=$(format_duration "${elapsed}") stage_eta=$(format_duration "${remaining}")"
  done
  wait "${pid}"
  rc=$?
  echo "STAGE_END=${stage} RC=${rc}"
  return "${rc}"
}

echo "EXPERIMENT_NAME=M2D-CLAP ${DATASET_LABEL} negative UIQ Tables 4 and 17"
echo "PAPER_EXPERIMENTS=EXP-16/17"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=M2D-CLAP"
echo "PROTOCOL_STATUS=controlled inferred deterministic released-caption pairing"
echo "DATASET=${DATASET_LABEL}; candidates=${EXPECTED_CANDIDATES}; negative_queries=${EXPECTED_QUERIES}"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "OEA_SOURCE_FILES=AudioRetrieval/models/m2d_clap_adapter.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py; AudioRetrieval/evaluation/negative_canonical.py"
echo "TOTAL_WORKLOAD=${EXPECTED_QUERIES} negative text embeddings; reuses ${EXPECTED_CANDIDATES} audio embeddings; canonical Table 17 metrics"
echo "ESTIMATED_TOTAL_TIME=2-10 minutes"
echo "DOWNLOADS_REQUIRED=no"
echo "OVERWRITE_DELETE_RISK=none; timestamped output"
echo "AUDIO_NPZ=${AUDIO_NPZ}"
echo "PAIRING_JSONL=${PAIRING_JSONL}"
echo "RESULT_DIRECTORY=${RESULT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

GIT_STATUS="$(git status --short | sed -E '/^\?\? scripts\/\.__dpc[[:xdigit:]]+$/d')"
[[ -z "${GIT_STATUS}" ]] || fail 3 "Git worktree is not clean"
for required in "${AUDIO_NPZ}" "${PAIRING_JSONL}" "${NEGATIVE_JSONL}" "${CHECKPOINT}" "${BERT_TOKENIZER}"; do
  [[ -e "${required}" ]] || fail 4 "Required resource is missing: ${required}"
done
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA256}" ]] \
  || fail 4 "Checkpoint SHA256 mismatch"
[[ "$(sha256sum "${PAIRING_JSONL}" | awk '{print $1}')" == "${PAIRING_SHA256}" ]] \
  || fail 4 "Pairing SHA256 mismatch"

printf '%s\n' "${GIT_COMMIT}" > "${RESULT_DIR}/git_commit.txt"
printf '%s\n' "${GIT_STATUS}" > "${RESULT_DIR}/git_status.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv \
  > "${RESULT_DIR}/gpu_info.txt" || fail 5 "nvidia-smi failed"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1' \
  || fail 5 "Exactly one visible CUDA GPU is required"

CURRENT_STAGE="gpu_negative_query_embeddings"
run_stage "${CURRENT_STAGE}" 480 \
  python -m AudioRetrieval preprocess uiq-embeddings \
    --model m2d_clap --dataset "${DATASET_LABEL}" --device cuda \
    --batch-size-text 128 --uiq-jsonl "${NEGATIVE_JSONL}" \
    --output-dir "${QUERY_DIR}" --m2d-ckpt "${CHECKPOINT}" \
    --m2d-bert-tokenizer "${BERT_TOKENIZER}" \
  || fail $? "M2D-CLAP negative query embedding generation failed"

CURRENT_STAGE="cpu_table17_metrics"
export CUDA_VISIBLE_DEVICES=""
run_stage "${CURRENT_STAGE}" 180 \
  python scripts/evaluate_negative_uiq_npz.py \
    --audio-npz "${AUDIO_NPZ}" \
    --query-npz "${QUERY_DIR}/uiq_negative_embeddings.npz" \
    --pairing-jsonl "${PAIRING_JSONL}" --output-dir "${METRICS_DIR}" \
    --model "M2D-CLAP" --dataset "${DATASET_LABEL}" \
    --expected-candidates "${EXPECTED_CANDIDATES}" \
    --expected-queries "${EXPECTED_QUERIES}" \
  || fail $? "M2D-CLAP Table 17 metric evaluation failed"

sha256sum "${QUERY_DIR}/uiq_negative_embeddings.npz" \
  "${METRICS_DIR}/metrics.json" "${LOG_FILE}" \
  > "${RESULT_DIR}/artifact_sha256.txt" \
  || fail $? "Final artifact hashing failed"

finish 0 complete none none
