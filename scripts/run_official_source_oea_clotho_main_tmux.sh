#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] This GPU experiment must run inside a visible tmux pane." >&2
  echo "[INFO] Create or attach a named tmux session, then run this script there." >&2
  exit 2
fi

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <oea_qwen7b|oea_qwen7b_cl>" >&2
  exit 2
fi

VARIANT_ID="$1"
case "${VARIANT_ID}" in
  oea_qwen7b)
    PAPER_MODEL="OEA-Qwen7B"
    RELEASED_MODEL="OEA-Qwen7B-AC"
    CHECKPOINT_SUBPATH="OEA-Qwen7B-AC/step_300.pt"
    ;;
  oea_qwen7b_cl)
    PAPER_MODEL="OEA-Qwen7B (+Cl)"
    RELEASED_MODEL="OEA-Qwen7B-Cl"
    CHECKPOINT_SUBPATH="OEA-Qwen7B-Cl/step_330.pt"
    ;;
  *)
    echo "[ERROR] This main-table runner accepts only the two remaining Qwen7B OEA variants." >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
BASE_MODEL_DIR="${MODEL_ROOT}/Qwen2.5-Omni-7B"
CHECKPOINT_PATH="${MODEL_ROOT}/${CHECKPOINT_SUBPATH}"
AUDIO_DIR="${DATA_ROOT}/clotho_v2.1/extracted/evaluation"
CAPTIONS_CSV="${DATA_ROOT}/clotho_v2.1/source/clotho_captions_evaluation.csv"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="official_source_${VARIANT_ID}_clotho_main_${RUN_STAMP}"
RESULT_ROOT="${ROOT_DIR}/results/raw/${RUN_ID}"
SMOKE_DIR="${RESULT_ROOT}/smoke"
EMBEDDING_DIR="${RESULT_ROOT}/embeddings"
METRICS_DIR="${RESULT_ROOT}/metrics"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"

if [[ -e "${RESULT_ROOT}" || -e "${LOG_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse a result or log directory." >&2
  echo "[ERROR] RESULT_ROOT=${RESULT_ROOT}" >&2
  echo "[ERROR] LOG_DIR=${LOG_DIR}" >&2
  exit 3
fi
mkdir -p "${SMOKE_DIR}" "${EMBEDDING_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_EPOCH="$(date +%s)"
START_TIME="$(date -Is)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --short)"
CURRENT_STAGE="preflight"

format_duration() {
  local total="${1:-0}"
  if (( total < 0 )); then total=0; fi
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
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
  metrics_path="${METRICS_DIR}/suite_metrics.json"
  printf '%s\n' "${rc}" > "${RESULT_ROOT}/exit_code.txt"
  echo "FINAL_RUN_RC=${rc}"
  echo "START_TIME=${START_TIME}"
  echo "END_TIME=${end_time}"
  echo "TOTAL_ELAPSED=$(format_duration "${elapsed}")"
  echo "COMPLETION_STATUS=${completion}"
  echo "RESULT_DIRECTORY=${RESULT_ROOT}"
  echo "LOG_DIRECTORY=${LOG_DIR}"
  echo "METRICS_PATH=${metrics_path}"
  echo "FAILED_STAGE=${failed_stage}"
  echo "ERROR_SUMMARY=${error_summary}"
  echo "TMUX_RETAINED_SHELL=yes"
  echo "[INFO] Experiment wrapper finished. The tmux pane remains open in an interactive shell."
  exec bash -i
}

if [[ -n "${GIT_STATUS}" ]]; then
  echo "[ERROR] Formal experiment requires a clean Git worktree." >&2
  printf '%s\n' "${GIT_STATUS}" >&2
  finish 3 failed preflight "Git worktree is not clean"
fi

for required in "${BASE_MODEL_DIR}" "${CHECKPOINT_PATH}" "${AUDIO_DIR}" "${CAPTIONS_CSV}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required resource is missing: ${required}" >&2
    finish 4 failed preflight "Required local resource is missing"
  fi
done

printf '%s\n' "${GIT_COMMIT}" > "${RESULT_ROOT}/git_commit.txt"
printf '%s\n' "${GIT_STATUS}" > "${RESULT_ROOT}/git_status.txt"

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

nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
  --format=csv > "${RESULT_ROOT}/gpu_info.txt"

echo "EXPERIMENT_NAME=${PAPER_MODEL} Clotho main Tables 2 and 3"
echo "PAPER_EXPERIMENTS=EXP-10 Table 2 T2A; EXP-11 Table 3 T2T"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS_SHORT=<clean>"
echo "MODEL=${PAPER_MODEL}; released checkpoint ${RELEASED_MODEL}"
echo "DATASET=Clotho v2.1 evaluation; 1,045 audio; 5,225 captions"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "OEA_OFFICIAL_SOURCE_FILES=examples/encode_example.py,AudioRetrieval/preprocessing/embeddings/oea.py,AudioRetrieval/evaluation/metrics.py"
echo "RESOURCES=1 CUDA GPU for smoke and embeddings; CPU-only four-protocol finalization"
echo "TOTAL_WORKLOAD=5-audio/5-query official smoke + 1,045 audio + 5,225 captions + 4 retrieval protocols"
echo "ESTIMATED_TOTAL_TIME=15-35 minutes"
echo "ETA_BASIS=Nemo3B official-source full embedding completed in approximately 6 minutes; Qwen7B memory pressure and model load add uncertainty"
echo "CACHE_DIRECTORY=${MODEL_ROOT} (read-only)"
echo "RESULT_DIRECTORY=${RESULT_ROOT}"
echo "LOG_FILE=${LOG_FILE}"
echo "METRICS_PATH=${METRICS_DIR}/suite_metrics.json"
echo "START_TIME=${START_TIME}"
echo "RESUME=use a new RUN_ID after failure; completed artifacts are never overwritten"

run_stage() {
  local stage="$1"
  local expected_seconds="$2"
  shift 2
  local stage_start pid now stage_elapsed overall_elapsed remaining finish_time
  CURRENT_STAGE="${stage}"
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
    if (( remaining > 0 )); then
      finish_time="$(date -d "@$((${now} + remaining))" -Is)"
    else
      finish_time="unknown"
    fi
    echo "PROGRESS stage=${stage} stage_elapsed=$(format_duration "${stage_elapsed}") overall_elapsed=$(format_duration "${overall_elapsed}") stage_eta=$(format_duration "${remaining}") overall_eta_basis=stage-estimate estimated_finish=${finish_time} throughput=see-live-tqdm-above"
  done
  wait "${pid}"
  local rc=$?
  echo "STAGE_END=${stage} RC=${rc}"
  return "${rc}"
}

run_stage official_source_smoke 480 \
  python examples/encode_example.py \
    --model "${RELEASED_MODEL}" \
    --checkpoint-file "$(basename "${CHECKPOINT_PATH}")" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --base-model-path "${BASE_MODEL_DIR}" \
    --device cuda \
  | tee "${SMOKE_DIR}/stdout.log"
SMOKE_RC=${PIPESTATUS[0]}
if [[ "${SMOKE_RC}" -ne 0 ]]; then
  finish "${SMOKE_RC}" failed official_source_smoke "Official source smoke failed; inspect combined.log"
fi

run_stage official_source_full_embeddings 1500 \
  python -m AudioRetrieval preprocess embeddings \
    --model oea \
    --audio-dir "${AUDIO_DIR}" \
    --captions-csv "${CAPTIONS_CSV}" \
    --output-dir "${EMBEDDING_DIR}" \
    --dataset clotho \
    --device cuda \
    --batch-size-audio 1 \
    --batch-size-text 16 \
    --checkpoint "${CHECKPOINT_PATH}" \
    --repo-id Qwen/Qwen2.5-Omni-7B \
    --local-path "${BASE_MODEL_DIR}"
EMBEDDING_RC=$?
if [[ "${EMBEDDING_RC}" -ne 0 ]]; then
  finish "${EMBEDDING_RC}" failed official_source_full_embeddings "Official source embedding precomputation failed; inspect combined.log"
fi

export CUDA_VISIBLE_DEVICES=""
run_stage cpu_table2_table3_finalizer 120 \
  python scripts/evaluate_official_source_oea_clotho.py \
    --baseline-dir "${EMBEDDING_DIR}" \
    --captions-csv "${CAPTIONS_CSV}" \
    --output-dir "${METRICS_DIR}" \
    --model "${PAPER_MODEL}" \
    --seed 0
METRIC_RC=$?
if [[ "${METRIC_RC}" -ne 0 ]]; then
  finish "${METRIC_RC}" failed cpu_table2_table3_finalizer "Table 2/3 metric finalization failed; inspect suite_metrics.json and combined.log"
fi

sha256sum \
  "${EMBEDDING_DIR}/audio_embeddings.npz" \
  "${EMBEDDING_DIR}/caption_embeddings.npz" \
  "${METRICS_DIR}/suite_metrics.json" \
  "${LOG_FILE}" \
  > "${RESULT_ROOT}/artifact_sha256.txt"

finish 0 complete none none
