#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
SOURCE_DIR="${MODEL_ROOT}/robust-clap/source"
SOURCE_REVISION="d08d0e3c545fa22df0930fc0d090741aaa9e2cc1"
CHECKPOINT="${MODEL_ROOT}/laion-clap/630k-audioset-best.pt"
CHECKPOINT_BYTES=1863587645
CHECKPOINT_SHA256="8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
BERT_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bert-base-uncased"
ROBERTA_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/roberta-base"
BART_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bart-base"
RUNTIME_OVERLAY="${MODEL_ROOT}/python/laion-clap-1.1.6-overlay-v3"
BPE_VOCAB="${RUNTIME_OVERLAY}/laion_clap/clap_module/bpe_simple_vocab_16e6.txt.gz"
BPE_VOCAB_BYTES=1356917
BPE_VOCAB_SHA256="924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
AUDIO_DIR="${DATA_ROOT}/clotho_v2.1/extracted/evaluation"
CAPTIONS_CSV="${DATA_ROOT}/clotho_v2.1/source/clotho_captions_evaluation.csv"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="robust_clap_clotho_main_${STAMP}"
RESULT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}"
SMOKE_DIR="${RESULT_DIR}/smoke"
EMBEDDING_DIR="${RESULT_DIR}/embeddings"
UIQ_DIR="${RESULT_DIR}/uiq_embeddings"
METRICS_DIR="${RESULT_DIR}/metrics"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"

if [[ -e "${RESULT_DIR}" || -e "${LOG_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse result or log directory." >&2
  exit 2
fi
mkdir -p "${SMOKE_DIR}" "${LOG_DIR}"
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
  echo "METRICS_PATH=${METRICS_DIR}/suite_metrics.json"
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

echo "EXPERIMENT_NAME=Robust-CLAP Clotho Tables 2/3/12-15 controlled binding"
echo "PAPER_EXPERIMENTS=EXP-10 Table 2; EXP-11 Table 3; EXP-12-15 positive UIQ"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=Robust-CLAP upstream source with SHA256-locked standard LAION checkpoint; HTSAT-tiny; RoBERTa; non-fusion"
echo "SOURCE_REVISION=${SOURCE_REVISION}"
echo "PROTOCOL_STATUS=controlled public-code reproduction; strict Robust-specific paper checkpoint identity not claimed"
echo "DATASET=Clotho v2.1 evaluation; 1,045 audio; 5,225 captions; 4,180 released UIQ queries"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "OEA_SOURCE_FILES=AudioRetrieval/models/robust_clap_adapter.py; AudioRetrieval/preprocessing/embeddings/robust_clap.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py; AudioRetrieval/evaluation/metrics.py"
echo "TOTAL_WORKLOAD=5 audio/25 caption smoke + 1,045 audio + 5,225 captions + 4,180 UIQ + 8 protocols"
echo "ESTIMATED_TOTAL_TIME=10-30 minutes"
echo "DOWNLOADS_REQUIRED=no"
echo "OVERWRITE_DELETE_RISK=none; timestamped result and log directories"
echo "RESULT_DIRECTORY=${RESULT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

GIT_STATUS="$(git status --short | sed -E '/^\?\? scripts\/\.__dpc[[:xdigit:]]+$/d')"
[[ -z "${GIT_STATUS}" ]] || fail 3 "Git worktree is not clean"
for required in "${SOURCE_DIR}" "${CHECKPOINT}" "${BERT_TOKENIZER}" \
  "${ROBERTA_TOKENIZER}" "${BART_TOKENIZER}" \
  "${RUNTIME_OVERLAY}" "${BPE_VOCAB}" "${AUDIO_DIR}" "${CAPTIONS_CSV}"; do
  [[ -e "${required}" ]] || fail 4 "Required resource is missing: ${required}"
done
python scripts/validate_robust_clap_source.py \
  --source-dir "${SOURCE_DIR}" --output "${RESULT_DIR}/source_identity.json" \
  || fail 4 "Robust-CLAP source identity mismatch"
[[ "$(stat -c %s "${CHECKPOINT}")" == "${CHECKPOINT_BYTES}" ]] \
  || fail 4 "Checkpoint byte size mismatch"
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA256}" ]] \
  || fail 4 "Checkpoint SHA256 mismatch"
[[ "$(stat -c %s "${BPE_VOCAB}")" == "${BPE_VOCAB_BYTES}" ]] \
  || fail 4 "BPE vocabulary byte size mismatch"
[[ "$(sha256sum "${BPE_VOCAB}" | awk '{print $1}')" == "${BPE_VOCAB_SHA256}" ]] \
  || fail 4 "BPE vocabulary SHA256 mismatch"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export PYTHONPATH="${RUNTIME_OVERLAY}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -c 'import laion_clap, torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1' \
  || fail 5 "Pinned runtime or exactly one visible CUDA GPU is unavailable"

python scripts/select_clotho_smoke_csv.py \
  --input "${CAPTIONS_CSV}" --output "${SMOKE_DIR}/captions.csv" --rows 5 \
  || fail $? "Smoke CSV selection failed"

ROBUST_ARGS=(
  --model robust_clap --device cuda
  --robust-ckpt "${CHECKPOINT}"
  --robust-repo "${SOURCE_DIR}"
  --robust-bert-tokenizer "${BERT_TOKENIZER}"
  --robust-roberta-tokenizer "${ROBERTA_TOKENIZER}"
  --robust-bart-tokenizer "${BART_TOKENIZER}"
  --robust-bpe-vocab "${BPE_VOCAB}"
)

CURRENT_STAGE="gpu_smoke_embeddings"
run_stage "${CURRENT_STAGE}" 600 \
  python -m AudioRetrieval preprocess embeddings "${ROBUST_ARGS[@]}" \
    --audio-dir "${AUDIO_DIR}" --captions-csv "${SMOKE_DIR}/captions.csv" \
    --output-dir "${SMOKE_DIR}" --dataset clotho \
    --batch-size-audio 5 --batch-size-text 25 \
  || fail $? "Robust-CLAP smoke embedding generation failed"
python scripts/validate_robust_clap_embeddings.py \
  --embedding-dir "${SMOKE_DIR}" --source-dir "${SOURCE_DIR}" \
  --checkpoint "${CHECKPOINT}" --expected-audio 5 --expected-captions 25 \
  --output "${SMOKE_DIR}/generation_metrics.json" \
  || fail $? "Robust-CLAP smoke artifact gate failed"

CURRENT_STAGE="gpu_full_embeddings"
run_stage "${CURRENT_STAGE}" 1200 \
  python -m AudioRetrieval preprocess embeddings "${ROBUST_ARGS[@]}" \
    --audio-dir "${AUDIO_DIR}" --captions-csv "${CAPTIONS_CSV}" \
    --output-dir "${EMBEDDING_DIR}" --dataset clotho \
    --batch-size-audio 32 --batch-size-text 128 \
  || fail $? "Robust-CLAP full embedding generation failed"
python scripts/validate_robust_clap_embeddings.py \
  --embedding-dir "${EMBEDDING_DIR}" --source-dir "${SOURCE_DIR}" \
  --checkpoint "${CHECKPOINT}" --expected-audio 1045 --expected-captions 5225 \
  --output "${EMBEDDING_DIR}/generation_metrics.json" \
  || fail $? "Robust-CLAP full artifact gate failed"

CURRENT_STAGE="gpu_uiq_embeddings"
run_stage "${CURRENT_STAGE}" 600 \
  python -m AudioRetrieval preprocess uiq-embeddings "${ROBUST_ARGS[@]}" \
    --dataset "Clotho v2.1 evaluation UIQ" --batch-size-text 128 \
    --uiq-jsonl data/UIQ/clotho/clotho_evaluation_question_queries.jsonl \
    --uiq-jsonl data/UIQ/clotho/clotho_evaluation_imperative_queries.jsonl \
    --uiq-jsonl data/UIQ/clotho/clotho_evaluation_paraphrase_queries.jsonl \
    --uiq-jsonl data/UIQ/clotho/clotho_evaluation_tagging_queries.jsonl \
    --output-dir "${UIQ_DIR}" \
  || fail $? "Robust-CLAP UIQ embedding generation failed"

CURRENT_STAGE="cpu_table2_table3_table12_table15_metrics"
export CUDA_VISIBLE_DEVICES=""
run_stage "${CURRENT_STAGE}" 300 \
  python scripts/evaluate_official_source_oea_clotho.py \
    --baseline-dir "${EMBEDDING_DIR}" --uiq-dir "${UIQ_DIR}" \
    --captions-csv "${CAPTIONS_CSV}" --output-dir "${METRICS_DIR}" \
    --model "Robust-CLAP controlled standard-checkpoint binding" --seed 0 \
  || fail $? "Robust-CLAP metric finalization failed"

sha256sum "${SMOKE_DIR}/generation_metrics.json" \
  "${EMBEDDING_DIR}/generation_metrics.json" \
  "${EMBEDDING_DIR}/audio_embeddings.npz" \
  "${EMBEDDING_DIR}/caption_embeddings.npz" \
  "${UIQ_DIR}"/uiq_*_embeddings.npz "${METRICS_DIR}/suite_metrics.json" \
  "${LOG_FILE}" > "${RESULT_DIR}/artifact_sha256.txt" \
  || fail $? "Final artifact hashing failed"

finish 0 complete none none
