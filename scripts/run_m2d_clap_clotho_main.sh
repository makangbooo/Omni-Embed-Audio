#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
CHECKPOINT="${MODEL_ROOT}/m2d-clap/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"
BERT_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bert-base-uncased"
AUDIO_DIR="${DATA_ROOT}/clotho_v2.1/extracted/evaluation"
CAPTIONS_CSV="${DATA_ROOT}/clotho_v2.1/source/clotho_captions_evaluation.csv"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="m2d_clap_clotho_main_${STAMP}"
RESULT_ROOT="${ROOT_DIR}/results/raw/${RUN_ID}"
SMOKE_DIR="${RESULT_ROOT}/smoke"
FULL_DIR="${RESULT_ROOT}/embeddings"
METRICS_DIR="${RESULT_ROOT}/metrics"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"
MODEL_LOCK="${RESULT_ROOT}/m2d_clap.portable_model_lock.json"

if [[ -e "${RESULT_ROOT}" || -e "${LOG_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse result or log directory." >&2
  exit 2
fi
mkdir -p "${SMOKE_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_TIME="$(date -Is)"
START_EPOCH="$(date +%s)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_STATUS_RAW="$(git status --short)"
GIT_STATUS="$(printf '%s\n' "${GIT_STATUS_RAW}" | \
  sed -E '/^\?\? scripts\/\.__dpc[[:xdigit:]]+$/d')"
GIT_STATUS_IGNORED="$(printf '%s\n' "${GIT_STATUS_RAW}" | \
  sed -nE '/^\?\? scripts\/\.__dpc[[:xdigit:]]+$/p')"

format_duration() {
  local total="${1:-0}"
  printf '%02d:%02d:%02d' \
    "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

finish() {
  local rc="$1" status="$2" stage="$3" summary="$4" elapsed
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
  exit "${rc}"
}

if [[ -n "${GIT_STATUS}" ]]; then
  printf '%s\n' "${GIT_STATUS}" >&2
  finish 3 failed preflight "Git worktree is not clean"
fi
for required in "${CHECKPOINT}" "${BERT_TOKENIZER}" "${AUDIO_DIR}" "${CAPTIONS_CSV}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required resource is missing: ${required}" >&2
    finish 4 failed preflight "Required resource is missing"
  fi
done

printf '%s\n' "${GIT_COMMIT}" > "${RESULT_ROOT}/git_commit.txt"
printf '%s\n' "${GIT_STATUS}" > "${RESULT_ROOT}/git_status.txt"
printf '%s\n' "${GIT_STATUS_IGNORED}" > "${RESULT_ROOT}/git_status_ignored.txt"
if [[ -n "${GIT_STATUS_IGNORED}" ]]; then
  printf 'IGNORED_EPHEMERAL_GIT_STATUS=%s\n' "${GIT_STATUS_IGNORED}"
fi
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
  --format=csv > "${RESULT_ROOT}/gpu_info.txt"

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

python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1'
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed preflight "Exactly one visible CUDA GPU is required"
fi

echo "EXPERIMENT_NAME=M2D-CLAP Clotho main Tables 2 and 3"
echo "PAPER_EXPERIMENTS=EXP-10 Table 2 T2A; EXP-11 Table 3 T2T"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=M2D-CLAP v0.5.0; checkpoint-30.pth; BERT-base text encoder"
echo "PROTOCOL_STATUS=controlled public-code reproduction"
echo "DATASET=Clotho v2.1 evaluation; 1,045 audio; 5,225 captions"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "TOTAL_WORKLOAD=5 audio/25 captions smoke + 1,045 audio + 5,225 captions + 4 protocols"
echo "ESTIMATED_TOTAL_TIME=15-40 minutes"
echo "RESULT_DIRECTORY=${RESULT_ROOT}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

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

run_stage resource_revalidation 240 \
  python scripts/build_m2d_clap_portable_lock.py \
    --model-root "${MODEL_ROOT}" --output "${MODEL_LOCK}"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed resource_revalidation "M2D resource validation failed"
fi

python scripts/select_clotho_smoke_csv.py \
  --input "${CAPTIONS_CSV}" --output "${SMOKE_DIR}/captions.csv" --rows 5
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed smoke_selection "Smoke CSV selection failed"
fi

run_stage gpu_smoke_embeddings 600 \
  python -m AudioRetrieval preprocess embeddings \
    --model m2d_clap --audio-dir "${AUDIO_DIR}" \
    --captions-csv "${SMOKE_DIR}/captions.csv" --output-dir "${SMOKE_DIR}" \
    --dataset clotho --device cuda --batch-size-audio 5 --batch-size-text 25 \
    --m2d-ckpt "${CHECKPOINT}" --m2d-bert-tokenizer "${BERT_TOKENIZER}"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed gpu_smoke_embeddings "M2D smoke embedding generation failed"
fi
python scripts/validate_m2d_clap_embeddings.py \
  --embedding-dir "${SMOKE_DIR}" --model-lock "${MODEL_LOCK}" \
  --expected-audio 5 --expected-captions 25 \
  --output "${SMOKE_DIR}/generation_metrics.json"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed smoke_validation "M2D smoke artifact gate failed"
fi

run_stage gpu_full_embeddings 1800 \
  python -m AudioRetrieval preprocess embeddings \
    --model m2d_clap --audio-dir "${AUDIO_DIR}" \
    --captions-csv "${CAPTIONS_CSV}" --output-dir "${FULL_DIR}" \
    --dataset clotho --device cuda --batch-size-audio 8 --batch-size-text 128 \
    --m2d-ckpt "${CHECKPOINT}" --m2d-bert-tokenizer "${BERT_TOKENIZER}"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed gpu_full_embeddings "M2D full embedding generation failed"
fi
python scripts/validate_m2d_clap_embeddings.py \
  --embedding-dir "${FULL_DIR}" --model-lock "${MODEL_LOCK}" \
  --expected-audio 1045 --expected-captions 5225 \
  --output "${FULL_DIR}/generation_metrics.json"
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed full_validation "M2D full artifact gate failed"
fi

export CUDA_VISIBLE_DEVICES=""
run_stage cpu_table2_table3_metrics 180 \
  python scripts/evaluate_official_source_oea_clotho.py \
    --baseline-dir "${FULL_DIR}" --captions-csv "${CAPTIONS_CSV}" \
    --output-dir "${METRICS_DIR}" --model "M2D-CLAP" --seed 0
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  finish "${RC}" failed cpu_table2_table3_metrics "M2D metric finalization failed"
fi

if ! sha256sum "${MODEL_LOCK}" "${SMOKE_DIR}/generation_metrics.json" \
  "${FULL_DIR}/generation_metrics.json" "${FULL_DIR}/audio_embeddings.npz" \
  "${FULL_DIR}/caption_embeddings.npz" "${METRICS_DIR}/suite_metrics.json" \
  "${LOG_FILE}" > "${RESULT_ROOT}/artifact_sha256.txt"; then
  finish 5 failed artifact_hashing "Final artifact hashing failed"
fi
finish 0 complete none none
