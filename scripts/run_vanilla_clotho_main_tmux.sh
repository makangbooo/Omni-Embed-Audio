#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "$#" -ne 1 || "$1" != "vanilla_qwen2_5_omni_7b" ]]; then
  echo "Usage: $0 vanilla_qwen2_5_omni_7b" >&2
  exit 2
fi
BACKBONE_ID="$1"

if [[ -z "${TMUX:-}" ]]; then
  SESSION_NAME="vanilla_qwen7b_main_$(date +%Y%m%d_%H%M%S)"
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[ERROR] tmux session already exists: ${SESSION_NAME}" >&2
    exit 3
  fi
  tmux new-session -d -s "${SESSION_NAME}" \
    "cd \"${ROOT_DIR}\" && exec bash scripts/run_vanilla_clotho_main_tmux.sh ${BACKBONE_ID}"
  echo "TMUX_SESSION=${SESSION_NAME}"
  echo "GPU_USED=yes"
  echo "OEA_OFFICIAL_SOURCE_USED=yes"
  echo "ESTIMATED_TOTAL_TIME=15-35 minutes"
  echo "ATTACH_COMMAND=tmux attach -t ${SESSION_NAME}"
  echo "DETACH_KEYS=Ctrl-b d"
  echo "CAPTURE_COMMAND=tmux capture-pane -pt ${SESSION_NAME} -S -120"
  exit 0
fi

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MODEL_DIR="${MODEL_ROOT}/Qwen2.5-Omni-7B"
MODEL_LOCK="${ROOT_DIR}/results/model_locks/vanilla_qwen2_5_omni_7b.json"
MANIFEST="${DATA_ROOT}/clotho_v2.1/manifests/clotho_evaluation_manifest.jsonl"
CAPTIONS_CSV="${DATA_ROOT}/clotho_v2.1/source/clotho_captions_evaluation.csv"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_ID="vanilla_qwen2_5_omni_7b_clotho_main_${RUN_STAMP}"
SMOKE_ID="vanilla_qwen2_5_omni_7b_clotho_smoke_seed42_${RUN_STAMP}"
FULL_ID="vanilla_qwen2_5_omni_7b_clotho_embeddings_seed42_${RUN_STAMP}"
SUITE_ID="vanilla_qwen2_5_omni_7b_clotho_retrieval_suite_seed42_${RUN_STAMP}"
MAIN_ROOT="${ROOT_DIR}/results/raw/${MAIN_ID}"
SMOKE_DIR="${MAIN_ROOT}/${SMOKE_ID}"
FULL_DIR="${MAIN_ROOT}/${FULL_ID}"
SUITE_DIR="${MAIN_ROOT}/${SUITE_ID}"
LOG_DIR="${ROOT_DIR}/logs/${MAIN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"

if [[ -e "${MAIN_ROOT}" || -e "${LOG_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse a result or log directory." >&2
  exit 4
fi
mkdir -p "${MAIN_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_EPOCH="$(date +%s)"
START_TIME="$(date -Is)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --short)"

format_duration() {
  local total="${1:-0}"
  if (( total < 0 )); then total=0; fi
  printf '%02d:%02d:%02d' \
    "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

finish() {
  local rc="$1"
  local completion="$2"
  local failed_stage="$3"
  local error_summary="$4"
  local end_epoch end_time elapsed metrics_path
  end_epoch="$(date +%s)"
  end_time="$(date -Is)"
  elapsed="$((end_epoch - START_EPOCH))"
  metrics_path="${SUITE_DIR}/suite_metrics.json"
  printf '%s\n' "${rc}" > "${MAIN_ROOT}/exit_code.txt"
  echo "FINAL_RUN_RC=${rc}"
  echo "START_TIME=${START_TIME}"
  echo "END_TIME=${end_time}"
  echo "TOTAL_ELAPSED_SECONDS=${elapsed}"
  echo "TOTAL_ELAPSED=$(format_duration "${elapsed}")"
  echo "COMPLETION_STATUS=${completion}"
  echo "FAILED_STAGE=${failed_stage}"
  echo "ERROR_SUMMARY=${error_summary}"
  echo "RESULT_DIRECTORY=${MAIN_ROOT}"
  echo "SMOKE_DIRECTORY=${SMOKE_DIR}"
  echo "FULL_DIRECTORY=${FULL_DIR}"
  echo "SUITE_DIRECTORY=${SUITE_DIR}"
  echo "LOG_DIRECTORY=${LOG_DIR}"
  echo "METRICS_PATH=${metrics_path}"
  echo "TMUX_RETAINED_SHELL=yes"
  exec bash -i
}

if [[ -n "${GIT_STATUS}" ]]; then
  printf '%s\n' "${GIT_STATUS}" >&2
  finish 5 failed preflight "Git worktree is not clean"
fi

for required in "${MODEL_DIR}" "${MODEL_LOCK}" "${MANIFEST}" "${CAPTIONS_CSV}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required resource is missing: ${required}" >&2
    finish 6 failed preflight "Required local resource is missing"
  fi
done

printf '%s\n' "${GIT_COMMIT}" > "${MAIN_ROOT}/git_commit.txt"
printf '%s\n' "${GIT_STATUS}" > "${MAIN_ROOT}/git_status.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
  --format=csv > "${MAIN_ROOT}/gpu_info.txt"

echo "EXPERIMENT_NAME=Vanilla Qwen2.5-Omni-7B Clotho main Tables 2 and 3"
echo "PAPER_EXPERIMENTS=EXP-10 Table 2 T2A; EXP-11 Table 3 T2T"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS_SHORT=<clean>"
echo "MODEL=Qwen2.5-Omni-7B base-only"
echo "MODEL_LOCK=${MODEL_LOCK}"
echo "DATASET=Clotho v2.1 evaluation; 1,045 audio; 5,225 captions"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "OEA_OFFICIAL_SOURCE_FILES=AudioRetrieval/models/omni_embed_adapter.py,AudioRetrieval/evaluation/metrics.py"
echo "BASE_ONLY_BOUNDARY=oea_checkpoint_loaded=false,lora_loaded=false,projection_head_loaded=false"
echo "RESOURCES=1 CUDA GPU for smoke/full embeddings; CPU-only four-protocol finalization"
echo "TOTAL_WORKLOAD=5 audio/25 captions smoke + 1,045 audio + 5,225 captions + 4 retrieval protocols"
echo "ESTIMATED_TOTAL_TIME=15-35 minutes"
echo "ETA_BASIS=Vanilla Qwen3B completed in 731 seconds; Qwen7B load and inference are larger"
echo "CACHE_DIRECTORY=${MODEL_ROOT} (read-only)"
echo "RESULT_DIRECTORY=${MAIN_ROOT}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

run_stage() {
  local stage="$1"
  local expected_seconds="$2"
  shift 2
  local stage_start pid now stage_elapsed overall_elapsed remaining
  stage_start="$(date +%s)"
  echo "STAGE_START=${stage}"
  "$@" &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 15
    now="$(date +%s)"
    stage_elapsed="$((now - stage_start))"
    overall_elapsed="$((now - START_EPOCH))"
    remaining="$((expected_seconds - stage_elapsed))"
    if (( remaining < 0 )); then remaining=0; fi
    echo "PROGRESS stage=${stage} stage_elapsed=$(format_duration "${stage_elapsed}") overall_elapsed=$(format_duration "${overall_elapsed}") stage_eta=$(format_duration "${remaining}") throughput=see-live-output"
  done
  wait "${pid}"
  local rc=$?
  echo "STAGE_END=${stage} RC=${rc}"
  return "${rc}"
}

run_stage gpu_smoke 600 \
  env RUN_ID="${SMOKE_ID}" RESULT_ROOT="${MAIN_ROOT}" \
  bash scripts/run_vanilla_backbone_embeddings.sh "${BACKBONE_ID}" --smoke
SMOKE_RC=$?
if [[ "${SMOKE_RC}" -ne 0 ]]; then
  finish "${SMOKE_RC}" failed gpu_smoke "Vanilla Qwen7B smoke failed"
fi

run_stage gpu_full_embeddings 1500 \
  env RUN_ID="${FULL_ID}" RESULT_ROOT="${MAIN_ROOT}" \
  bash scripts/run_vanilla_backbone_embeddings.sh \
    "${BACKBONE_ID}" --full "${SMOKE_DIR}/generation_metrics.json"
FULL_RC=$?
if [[ "${FULL_RC}" -ne 0 ]]; then
  finish "${FULL_RC}" failed gpu_full_embeddings "Vanilla Qwen7B full embedding generation failed"
fi

run_stage cpu_table2_table3_metrics 180 \
  env SUITE_ID="${SUITE_ID}" RESULT_ROOT="${MAIN_ROOT}" \
  bash scripts/run_vanilla_clotho_retrieval_suite.sh \
    "${BACKBONE_ID}" "${FULL_DIR}"
SUITE_RC=$?
if [[ "${SUITE_RC}" -ne 0 ]]; then
  finish "${SUITE_RC}" failed cpu_table2_table3_metrics "Vanilla Qwen7B retrieval suite failed"
fi

sha256sum \
  "${SMOKE_DIR}/generation_metrics.json" \
  "${FULL_DIR}/generation_metrics.json" \
  "${FULL_DIR}/candidate_embeddings.npy" \
  "${FULL_DIR}/query_embeddings.npy" \
  "${SUITE_DIR}/suite_metrics.json" \
  "${SUITE_DIR}/retrieval_summary.csv" \
  > "${MAIN_ROOT}/artifact_sha256.txt"

finish 0 complete none none
