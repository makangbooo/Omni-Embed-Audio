#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <laion-resource-pipeline-directory>" >&2
  exit 2
fi
RESOURCE_RUN_DIR="$(readlink -f "$1")"
LOGS_ROOT="$(readlink -f "${ROOT_DIR}/logs")"
if [[ "${RESOURCE_RUN_DIR}" != "${LOGS_ROOT}"/* ]]; then
  echo "[ERROR] Resource pipeline directory must be below repository logs/." >&2
  exit 3
fi

if [[ -z "${TMUX:-}" ]]; then
  SESSION_NAME="laion_clap_clotho_main_$(date +%Y%m%d_%H%M%S)"
  tmux new-session -d -s "${SESSION_NAME}" \
    "cd \"${ROOT_DIR}\" && exec bash scripts/run_laion_clap_clotho_main_tmux.sh \"${RESOURCE_RUN_DIR}\""
  echo "TMUX_SESSION=${SESSION_NAME}"
  echo "GPU_USED=yes"
  echo "OEA_OFFICIAL_SOURCE_USED=yes"
  echo "ESTIMATED_TOTAL_TIME=10-25 minutes"
  echo "ATTACH_COMMAND=tmux attach -t ${SESSION_NAME}"
  echo "CAPTURE_COMMAND=tmux capture-pane -pt ${SESSION_NAME} -S -160"
  exit 0
fi

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MANIFEST="${ROOT_DIR}/configs/resources/model05_laion_clap.json"
REQUIREMENTS="${ROOT_DIR}/configs/resources/laion_clap_1_1_6_overlay.requirements.txt"
OVERLAY_ROOT="${MODEL_ROOT}/python/laion-clap-1.1.6-overlay"
DOWNLOAD_REPORT="${RESOURCE_RUN_DIR}/download_manifest.json"
RESOURCE_LOCK="${RESOURCE_RUN_DIR}/laion_clap.portable_model_lock.json"
CHECKPOINT="${MODEL_ROOT}/laion-clap/630k-audioset-best.pt"
BERT_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bert-base-uncased"
ROBERTA_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/roberta-base"
BART_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bart-base"
AUDIO_DIR="${DATA_ROOT}/clotho_v2.1/extracted/evaluation"
CAPTIONS_CSV="${DATA_ROOT}/clotho_v2.1/source/clotho_captions_evaluation.csv"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="laion_clap_clotho_main_${STAMP}"
RESULT_ROOT="${ROOT_DIR}/results/raw/${RUN_ID}"
SMOKE_DIR="${RESULT_ROOT}/smoke"
FULL_DIR="${RESULT_ROOT}/embeddings"
METRICS_DIR="${RESULT_ROOT}/metrics"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"
VALIDATED_LOCK="${RESULT_ROOT}/validated_model_lock.json"

if [[ -e "${RESULT_ROOT}" || -e "${LOG_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse result or log directory." >&2
  exit 4
fi
mkdir -p "${SMOKE_DIR}" "${FULL_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_TIME="$(date -Is)"
START_EPOCH="$(date +%s)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --short)"

format_duration() {
  local total="${1:-0}"
  printf '%02d:%02d:%02d' \
    "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

finish() {
  local rc="$1"
  local status="$2"
  local stage="$3"
  local summary="$4"
  local elapsed
  elapsed="$(($(date +%s) - START_EPOCH))"
  printf '%s\n' "${rc}" > "${RESULT_ROOT}/exit_code.txt"
  echo "FINAL_RUN_RC=${rc}"
  echo "START_TIME=${START_TIME}"
  echo "END_TIME=$(date -Is)"
  echo "TOTAL_ELAPSED_SECONDS=${elapsed}"
  echo "TOTAL_ELAPSED=$(format_duration "${elapsed}")"
  echo "COMPLETION_STATUS=${status}"
  echo "FAILED_STAGE=${stage}"
  echo "ERROR_SUMMARY=${summary}"
  echo "RESULT_DIRECTORY=${RESULT_ROOT}"
  echo "LOG_DIRECTORY=${LOG_DIR}"
  echo "METRICS_PATH=${METRICS_DIR}/suite_metrics.json"
  echo "TMUX_RETAINED_SHELL=yes"
  exec bash -i
}

if [[ -n "${GIT_STATUS}" ]]; then
  printf '%s\n' "${GIT_STATUS}" >&2
  finish 5 failed preflight "Git worktree is not clean"
fi
for required in \
  "${DOWNLOAD_REPORT}" "${RESOURCE_LOCK}" "${OVERLAY_ROOT}" \
  "${CHECKPOINT}" "${BERT_TOKENIZER}" "${ROBERTA_TOKENIZER}" \
  "${BART_TOKENIZER}" "${AUDIO_DIR}" "${CAPTIONS_CSV}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required resource is missing: ${required}" >&2
    finish 6 failed preflight "Required resource is missing"
  fi
done

printf '%s\n' "${GIT_COMMIT}" > "${RESULT_ROOT}/git_commit.txt"
printf '%s\n' "${GIT_STATUS}" > "${RESULT_ROOT}/git_status.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
  --format=csv > "${RESULT_ROOT}/gpu_info.txt"

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

python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1'
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed preflight "Exactly one visible CUDA GPU is required"
fi

echo "EXPERIMENT_NAME=LAION-CLAP Clotho main Tables 2 and 3"
echo "PAPER_EXPERIMENTS=EXP-10 Table 2 T2A; EXP-11 Table 3 T2T"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=LAION-CLAP 1.1.6; HTSAT-tiny; RoBERTa; non-fusion 630k-audioset-best.pt"
echo "PROTOCOL_STATUS=controlled public-code reproduction"
echo "DATASET=Clotho v2.1 evaluation; 1,045 audio; 5,225 captions"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "OEA_OFFICIAL_SOURCE_FILES=AudioRetrieval/models/laion_clap_adapter.py,AudioRetrieval/preprocessing/embeddings/laion_clap.py,AudioRetrieval/evaluation/metrics.py"
echo "TOTAL_WORKLOAD=5 audio/25 captions smoke + 1,045 audio + 5,225 captions + 4 protocols"
echo "ESTIMATED_TOTAL_TIME=10-25 minutes"
echo "RESULT_DIRECTORY=${RESULT_ROOT}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

run_stage() {
  local stage="$1"
  local expected_seconds="$2"
  shift 2
  local stage_start pid now elapsed remaining
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
  local rc=$?
  echo "STAGE_END=${stage} RC=${rc}"
  return "${rc}"
}

run_stage resource_revalidation 120 \
  python scripts/build_laion_clap_portable_lock.py \
    --model-root "${MODEL_ROOT}" \
    --overlay-root "${OVERLAY_ROOT}" \
    --download-report "${DOWNLOAD_REPORT}" \
    --manifest "${MANIFEST}" \
    --requirements "${REQUIREMENTS}" \
    --output "${VALIDATED_LOCK}"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed resource_revalidation "Resource revalidation failed"
fi

python scripts/select_clotho_smoke_csv.py \
  --input "${CAPTIONS_CSV}" --output "${SMOKE_DIR}/captions.csv" --rows 5
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed smoke_selection "Smoke CSV selection failed"
fi
run_stage gpu_smoke_embeddings 600 \
  python -m AudioRetrieval preprocess embeddings \
    --model laion_clap --audio-dir "${AUDIO_DIR}" \
    --captions-csv "${SMOKE_DIR}/captions.csv" --output-dir "${SMOKE_DIR}" \
    --dataset clotho --device cuda --batch-size-audio 5 --batch-size-text 25 \
    --laion-ckpt "${CHECKPOINT}" \
    --laion-bert-tokenizer "${BERT_TOKENIZER}" \
    --laion-roberta-tokenizer "${ROBERTA_TOKENIZER}" \
    --laion-bart-tokenizer "${BART_TOKENIZER}"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed gpu_smoke_embeddings "LAION-CLAP original smoke failed"
fi
python scripts/validate_laion_clap_embeddings.py \
  --embedding-dir "${SMOKE_DIR}" --model-lock "${VALIDATED_LOCK}" \
  --expected-audio 5 --expected-captions 25 \
  --output "${SMOKE_DIR}/generation_metrics.json"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed smoke_validation "Smoke artifact gate failed"
fi

run_stage gpu_full_embeddings 1200 \
  python -m AudioRetrieval preprocess embeddings \
    --model laion_clap --audio-dir "${AUDIO_DIR}" \
    --captions-csv "${CAPTIONS_CSV}" --output-dir "${FULL_DIR}" \
    --dataset clotho --device cuda --batch-size-audio 32 --batch-size-text 256 \
    --laion-ckpt "${CHECKPOINT}" \
    --laion-bert-tokenizer "${BERT_TOKENIZER}" \
    --laion-roberta-tokenizer "${ROBERTA_TOKENIZER}" \
    --laion-bart-tokenizer "${BART_TOKENIZER}"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed gpu_full_embeddings "Full LAION-CLAP embedding generation failed"
fi
python scripts/validate_laion_clap_embeddings.py \
  --embedding-dir "${FULL_DIR}" --model-lock "${VALIDATED_LOCK}" \
  --expected-audio 1045 --expected-captions 5225 \
  --output "${FULL_DIR}/generation_metrics.json"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed full_validation "Full artifact gate failed"
fi

export CUDA_VISIBLE_DEVICES=""
run_stage cpu_table2_table3_metrics 180 \
  python scripts/evaluate_official_source_oea_clotho.py \
    --baseline-dir "${FULL_DIR}" --captions-csv "${CAPTIONS_CSV}" \
    --output-dir "${METRICS_DIR}" --model "LAION-CLAP" --seed 0
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed cpu_table2_table3_metrics "Metric finalization failed"
fi

sha256sum \
  "${VALIDATED_LOCK}" "${SMOKE_DIR}/generation_metrics.json" \
  "${FULL_DIR}/generation_metrics.json" "${FULL_DIR}/audio_embeddings.npz" \
  "${FULL_DIR}/caption_embeddings.npz" "${METRICS_DIR}/suite_metrics.json" \
  "${LOG_FILE}" > "${RESULT_ROOT}/artifact_sha256.txt"
finish 0 complete none none
