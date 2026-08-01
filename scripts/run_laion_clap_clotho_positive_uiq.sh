#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <completed-laion-clap-clotho-main-result-directory>" >&2
  exit 2
fi

MAIN_DIR="$(cd "$1" && pwd)"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
OVERLAY_ROOT="${MODEL_ROOT}/python/laion-clap-1.1.6-overlay-v3"
CHECKPOINT="${MODEL_ROOT}/laion-clap/630k-audioset-best.pt"
BERT_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bert-base-uncased"
ROBERTA_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/roberta-base"
BART_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bart-base"
CAPTIONS_CSV="${DATA_ROOT}/clotho_v2.1/source/clotho_captions_evaluation.csv"
MODEL_LOCK="${MAIN_DIR}/validated_model_lock.json"
BASELINE_DIR="${MAIN_DIR}/embeddings"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="laion_clap_clotho_positive_uiq_${STAMP}"
RESULT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}"
UIQ_DIR="${RESULT_DIR}/uiq_embeddings"
METRICS_DIR="${RESULT_DIR}/metrics"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"

if [[ -e "${RESULT_DIR}" || -e "${LOG_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse result or log directory." >&2
  exit 2
fi
mkdir -p "${UIQ_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_TIME="$(date -Is)"
START_EPOCH="$(date +%s)"
GIT_COMMIT="$(git rev-parse HEAD)"
CURRENT_STAGE="preflight"

format_duration() {
  local total="${1:-0}"
  printf '%02d:%02d:%02d' \
    "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
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
  echo "METRICS_PATH=${METRICS_DIR}/suite_metrics.json"
  return "${rc}"
}

fail() {
  local rc="$1" summary="$2"
  finish "${rc}" failed "${CURRENT_STAGE}" "${summary}"
  exit "${rc}"
}

echo "EXPERIMENT_NAME=LAION-CLAP Clotho positive UIQ Tables 12-15"
echo "PAPER_EXPERIMENTS=EXP-12 Table 12 Question; EXP-13 Table 13 Imperative; EXP-14 Table 14 Paraphrase; EXP-15 Table 15 Keyphrase"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=LAION-CLAP 1.1.6; HTSAT-tiny; RoBERTa; non-fusion 630k-audioset-best.pt"
echo "PROTOCOL_STATUS=controlled public-code reproduction"
echo "DATASET=Clotho v2.1 evaluation; 1,045 audio candidates; 4,180 released UIQ queries"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "OEA_SOURCE_FILES=AudioRetrieval/models/laion_clap_adapter.py; AudioRetrieval/models/laion_clap_tokenizers.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py; scripts/evaluate_official_source_oea_clotho.py"
echo "TOTAL_WORKLOAD=4,180 UIQ text embeddings + 4 retrieval protocols; reuses completed audio embeddings"
echo "ESTIMATED_TOTAL_TIME=3-10 minutes"
echo "RESULT_DIRECTORY=${RESULT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

GIT_STATUS="$(git status --short | sed -E '/^\?\? scripts\/\.__dpc[[:xdigit:]]+$/d')"
if [[ -n "${GIT_STATUS}" ]]; then
  printf '%s\n' "${GIT_STATUS}" >&2
  fail 3 "Git worktree is not clean"
fi

for required in "${OVERLAY_ROOT}" "${CHECKPOINT}" "${BERT_TOKENIZER}" \
  "${ROBERTA_TOKENIZER}" "${BART_TOKENIZER}" "${CAPTIONS_CSV}" \
  "${MODEL_LOCK}" "${MAIN_DIR}/exit_code.txt" \
  "${BASELINE_DIR}/audio_embeddings.npz" \
  "${BASELINE_DIR}/caption_embeddings.npz"; do
  [[ -e "${required}" ]] || fail 4 "Required resource is missing: ${required}"
done

if [[ "$(<"${MAIN_DIR}/exit_code.txt")" != "0" ]]; then
  fail 4 "The supplied LAION-CLAP main run is not complete"
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export PYTHONPATH="${OVERLAY_ROOT}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1' \
  || fail 5 "Exactly one visible CUDA GPU is required"

CURRENT_STAGE="gpu_uiq_embeddings"
echo "STAGE_START=${CURRENT_STAGE}"
python -m AudioRetrieval preprocess uiq-embeddings \
  --model laion_clap \
  --dataset "Clotho v2.1 evaluation UIQ" \
  --device cuda \
  --batch-size-text 128 \
  --laion-ckpt "${CHECKPOINT}" \
  --laion-bert-tokenizer "${BERT_TOKENIZER}" \
  --laion-roberta-tokenizer "${ROBERTA_TOKENIZER}" \
  --laion-bart-tokenizer "${BART_TOKENIZER}" \
  --uiq-jsonl data/UIQ/clotho/clotho_evaluation_question_queries.jsonl \
  --uiq-jsonl data/UIQ/clotho/clotho_evaluation_imperative_queries.jsonl \
  --uiq-jsonl data/UIQ/clotho/clotho_evaluation_paraphrase_queries.jsonl \
  --uiq-jsonl data/UIQ/clotho/clotho_evaluation_tagging_queries.jsonl \
  --output-dir "${UIQ_DIR}" \
  || fail $? "LAION-CLAP UIQ embedding generation failed"
echo "STAGE_END=${CURRENT_STAGE} RC=0"

CURRENT_STAGE="cpu_table12_table15_metrics"
export CUDA_VISIBLE_DEVICES=""
echo "STAGE_START=${CURRENT_STAGE}"
python scripts/evaluate_official_source_oea_clotho.py \
  --baseline-dir "${BASELINE_DIR}" \
  --uiq-dir "${UIQ_DIR}" \
  --captions-csv "${CAPTIONS_CSV}" \
  --output-dir "${METRICS_DIR}" \
  --model "LAION-CLAP" \
  --seed 0 \
  || fail $? "LAION-CLAP positive UIQ metric finalization failed"
echo "STAGE_END=${CURRENT_STAGE} RC=0"

sha256sum "${MODEL_LOCK}" "${UIQ_DIR}"/uiq_*_embeddings.npz \
  "${METRICS_DIR}/suite_metrics.json" "${LOG_FILE}" \
  > "${RESULT_DIR}/artifact_sha256.txt" \
  || fail $? "Final artifact hashing failed"

finish 0 complete none none
