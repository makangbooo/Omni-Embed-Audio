#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <laion_clap|mga_clap|robust_clap> <clotho|audiocaps|mecat> <audio-embeddings.npz> <pairing-audit-root>" >&2
  exit 2
fi

MODEL_ID="$1"
DATASET_ID="$2"
AUDIO_NPZ="$(realpath "$3")"
PAIRING_ROOT="$(realpath "$4")"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
BERT_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bert-base-uncased"
ROBERTA_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/roberta-base"
BART_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bart-base"
LAION_OVERLAY="${MODEL_ROOT}/python/laion-clap-1.1.6-overlay-v3"

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

MODEL_ARGS=()
REQUIRED_MODEL_RESOURCES=()
case "${MODEL_ID}" in
  laion_clap)
    MODEL_LABEL="LAION-CLAP"
    CHECKPOINT="${MODEL_ROOT}/laion-clap/630k-audioset-best.pt"
    CHECKPOINT_SHA256="8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
    PYTHON_PREFIX="${LAION_OVERLAY}"
    PRECHECK_IMPORT="import laion_clap, torch"
    OEA_SOURCE_FILES="AudioRetrieval/models/laion_clap_adapter.py; AudioRetrieval/models/laion_clap_tokenizers.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py; AudioRetrieval/evaluation/negative_canonical.py"
    PROTOCOL_MODEL_BOUNDARY="pinned official public checkpoint"
    REQUIRED_MODEL_RESOURCES=(
      "${LAION_OVERLAY}" "${CHECKPOINT}" "${BERT_TOKENIZER}"
      "${ROBERTA_TOKENIZER}" "${BART_TOKENIZER}"
    )
    MODEL_ARGS=(
      --model laion_clap --laion-ckpt "${CHECKPOINT}"
      --laion-bert-tokenizer "${BERT_TOKENIZER}"
      --laion-roberta-tokenizer "${ROBERTA_TOKENIZER}"
      --laion-bart-tokenizer "${BART_TOKENIZER}"
    )
    ;;
  mga_clap)
    MODEL_LABEL="MGA-CLAP"
    SOURCE_DIR="${MODEL_ROOT}/mga-clap/source"
    CHECKPOINT="${MODEL_ROOT}/mga-clap/pretrained_models/models/model.pt"
    CHECKPOINT_SHA256="8703740b738e973a5b4d8a18a074ad56880e98f8ba21cd618d7d7ee5422d6e26"
    MGA_DEPENDENCY_OVERLAY="${MGA_DEPENDENCY_OVERLAY:-${MODEL_ROOT}/python/mga-clap-runtime-v1}"
    PYTHON_PREFIX="${MGA_DEPENDENCY_OVERLAY}:${LAION_OVERLAY}"
    PRECHECK_IMPORT="import torch"
    OEA_SOURCE_FILES="AudioRetrieval/models/mga_clap_adapter.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py; AudioRetrieval/evaluation/negative_canonical.py"
    PROTOCOL_MODEL_BOUNDARY="pinned official source and Google Drive checkpoint"
    REQUIRED_MODEL_RESOURCES=(
      "${SOURCE_DIR}" "${LAION_OVERLAY}" "${MGA_DEPENDENCY_OVERLAY}"
      "${CHECKPOINT}" "${BERT_TOKENIZER}"
    )
    MODEL_ARGS=(
      --model mga_clap --mga-repo "${SOURCE_DIR}" --mga-ckpt "${CHECKPOINT}"
      --mga-bert-tokenizer "${BERT_TOKENIZER}"
      --mga-checkpoint-sha256 "${CHECKPOINT_SHA256}"
    )
    ;;
  robust_clap)
    MODEL_LABEL="Robust-CLAP"
    SOURCE_DIR="${MODEL_ROOT}/robust-clap/source"
    CHECKPOINT="${MODEL_ROOT}/laion-clap/630k-audioset-best.pt"
    CHECKPOINT_SHA256="8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
    BPE_VOCAB="${LAION_OVERLAY}/laion_clap/clap_module/bpe_simple_vocab_16e6.txt.gz"
    BPE_VOCAB_SHA256="924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
    PYTHON_PREFIX="${LAION_OVERLAY}"
    PRECHECK_IMPORT="import laion_clap, torch"
    OEA_SOURCE_FILES="AudioRetrieval/models/robust_clap_adapter.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py; AudioRetrieval/evaluation/negative_canonical.py"
    PROTOCOL_MODEL_BOUNDARY="upstream Robust-CLAP source with pinned standard LAION checkpoint; strict paper checkpoint identity not claimed"
    REQUIRED_MODEL_RESOURCES=(
      "${SOURCE_DIR}" "${LAION_OVERLAY}" "${CHECKPOINT}"
      "${BERT_TOKENIZER}" "${ROBERTA_TOKENIZER}" "${BART_TOKENIZER}"
      "${BPE_VOCAB}"
    )
    MODEL_ARGS=(
      --model robust_clap --robust-ckpt "${CHECKPOINT}"
      --robust-repo "${SOURCE_DIR}"
      --robust-bert-tokenizer "${BERT_TOKENIZER}"
      --robust-roberta-tokenizer "${ROBERTA_TOKENIZER}"
      --robust-bart-tokenizer "${BART_TOKENIZER}"
      --robust-bpe-vocab "${BPE_VOCAB}"
    )
    ;;
  *)
    echo "Unsupported model: ${MODEL_ID}" >&2
    exit 2
    ;;
esac

PAIRING_JSONL="${PAIRING_ROOT}/${DATASET_ID}/pairing_metadata.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${MODEL_ID}_${DATASET_ID}_negative_uiq_${STAMP}"
RESULT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}"
QUERY_DIR="${RESULT_DIR}/query_embeddings"
METRICS_DIR="${RESULT_DIR}/metrics"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"

if [[ -e "${RESULT_DIR}" || -e "${LOG_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse result or log directory." >&2
  exit 2
fi
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

echo "EXPERIMENT_NAME=${MODEL_LABEL} ${DATASET_LABEL} negative UIQ Tables 4 and 17"
echo "PAPER_EXPERIMENTS=EXP-16/17"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=${MODEL_LABEL}; ${PROTOCOL_MODEL_BOUNDARY}"
echo "PROTOCOL_STATUS=controlled inferred deterministic released-caption pairing"
echo "DATASET=${DATASET_LABEL}; candidates=${EXPECTED_CANDIDATES}; negative_queries=${EXPECTED_QUERIES}"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "OEA_SOURCE_FILES=${OEA_SOURCE_FILES}"
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
for required in "${AUDIO_NPZ}" "${PAIRING_JSONL}" "${NEGATIVE_JSONL}" \
  "${REQUIRED_MODEL_RESOURCES[@]}"; do
  [[ -e "${required}" ]] || fail 4 "Required resource is missing: ${required}"
done
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA256}" ]] \
  || fail 4 "Checkpoint SHA256 mismatch"
[[ "$(sha256sum "${PAIRING_JSONL}" | awk '{print $1}')" == "${PAIRING_SHA256}" ]] \
  || fail 4 "Pairing SHA256 mismatch"
if [[ "${MODEL_ID}" == "robust_clap" ]]; then
  python scripts/validate_robust_clap_source.py \
    --source-dir "${SOURCE_DIR}" --output "${RESULT_DIR}/source_identity.json" \
    || fail 4 "Robust-CLAP source identity mismatch"
  [[ "$(sha256sum "${BPE_VOCAB}" | awk '{print $1}')" == "${BPE_VOCAB_SHA256}" ]] \
    || fail 4 "BPE vocabulary SHA256 mismatch"
fi

printf '%s\n' "${GIT_COMMIT}" > "${RESULT_DIR}/git_commit.txt"
printf '%s\n' "${GIT_STATUS}" > "${RESULT_DIR}/git_status.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv \
  > "${RESULT_DIR}/gpu_info.txt" || fail 5 "nvidia-smi failed"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export PYTHONPATH="${PYTHON_PREFIX}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -c "${PRECHECK_IMPORT}; assert torch.cuda.is_available() and torch.cuda.device_count() == 1" \
  || fail 5 "Pinned runtime or exactly one visible CUDA GPU is unavailable"

CURRENT_STAGE="gpu_negative_query_embeddings"
run_stage "${CURRENT_STAGE}" 480 \
  python -m AudioRetrieval preprocess uiq-embeddings \
    "${MODEL_ARGS[@]}" --dataset "${DATASET_LABEL}" --device cuda \
    --batch-size-text 128 --uiq-jsonl "${NEGATIVE_JSONL}" \
    --output-dir "${QUERY_DIR}" \
  || fail $? "${MODEL_LABEL} negative query embedding generation failed"

CURRENT_STAGE="cpu_table17_metrics"
export CUDA_VISIBLE_DEVICES=""
run_stage "${CURRENT_STAGE}" 180 \
  python scripts/evaluate_negative_uiq_npz.py \
    --audio-npz "${AUDIO_NPZ}" \
    --query-npz "${QUERY_DIR}/uiq_negative_embeddings.npz" \
    --pairing-jsonl "${PAIRING_JSONL}" --output-dir "${METRICS_DIR}" \
    --model "${MODEL_LABEL}" --dataset "${DATASET_LABEL}" \
    --expected-candidates "${EXPECTED_CANDIDATES}" \
    --expected-queries "${EXPECTED_QUERIES}" \
  || fail $? "${MODEL_LABEL} Table 17 metric evaluation failed"

sha256sum "${QUERY_DIR}/uiq_negative_embeddings.npz" \
  "${METRICS_DIR}/metrics.json" "${LOG_FILE}" \
  > "${RESULT_DIR}/artifact_sha256.txt" \
  || fail $? "Final artifact hashing failed"

finish 0 complete none none
