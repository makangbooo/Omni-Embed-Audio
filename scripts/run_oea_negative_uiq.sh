#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <oea_nemo3b|oea_nemo3b_cl|oea_qwen3b|oea_qwen3b_cl|oea_qwen7b|oea_qwen7b_cl> <pairing-audit-root>" >&2
  exit 2
fi

VARIANT_ID="$1"
PAIRING_ROOT="$(realpath "$2")"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"

case "${VARIANT_ID}" in
  oea_nemo3b)
    PAPER_MODEL="OEA-Nemo3B"
    REPO_ID="nvidia/omni-embed-nemotron-3b"
    BASE_MODEL_DIR="${MODEL_ROOT}/omni-embed-nemotron-3b"
    CHECKPOINT="${MODEL_ROOT}/OEA-Nemo3B-AC/step_400_best.pt"
    CHECKPOINT_BYTES=9466826153
    CHECKPOINT_SHA256="55579dfbd4f6621b5d842c5e731d6a1d37dbfd04b26b1c55bf8cdea980e67d25"
    CLOTHO_AUDIO_SOURCE="${ROOT_DIR}/results/raw/official_source_oea_nemo3b_ac_5clip_20260729_213846/baseline/audio_embeddings.npz"
    AUDIOCAPS_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_nemo3b_audiocaps_main_20260802_184302/embeddings/audio_embeddings.npz"
    MECAT_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_nemo3b_mecat_positive_uiq_public848_20260802_210656/audio_embeddings/audio_embeddings.npz"
    ;;
  oea_nemo3b_cl)
    PAPER_MODEL="OEA-Nemo3B (+Cl)"
    REPO_ID="nvidia/omni-embed-nemotron-3b"
    BASE_MODEL_DIR="${MODEL_ROOT}/omni-embed-nemotron-3b"
    CHECKPOINT="${MODEL_ROOT}/OEA-Nemo3B-Cl/step_450_best_inference_only.pt"
    CHECKPOINT_BYTES=59072047
    CHECKPOINT_SHA256="2a5bee9039a28c0028cf205d1e2f4302fda540dd913f4cc1b08301edfc6680c4"
    CLOTHO_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_nemo3b_clotho_embeddings_seed42_20260727_094554"
    AUDIOCAPS_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_nemo3b_cl_audiocaps_main_20260802_185113/embeddings/audio_embeddings.npz"
    MECAT_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_nemo3b_cl_mecat_positive_uiq_public848_20260802_211339/audio_embeddings/audio_embeddings.npz"
    ;;
  oea_qwen3b)
    PAPER_MODEL="OEA-Qwen3B"
    REPO_ID="Qwen/Qwen2.5-Omni-3B"
    BASE_MODEL_DIR="${MODEL_ROOT}/Qwen2.5-Omni-3B"
    CHECKPOINT="${MODEL_ROOT}/OEA-Qwen3B-AC/step_350_inference_only.pt"
    CHECKPOINT_BYTES=59069203
    CHECKPOINT_SHA256="b1d0f559711b70f5a80dbdeb8cd46d80ed7b38b8e5524b9a871878d8bcd5f101"
    CLOTHO_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_qwen3b_ac_clotho_embeddings_seed42_20260719_231724"
    AUDIOCAPS_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_qwen3b_audiocaps_main_20260802_185757/embeddings/audio_embeddings.npz"
    MECAT_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_qwen3b_mecat_positive_uiq_public848_20260802_211837/audio_embeddings/audio_embeddings.npz"
    ;;
  oea_qwen3b_cl)
    PAPER_MODEL="OEA-Qwen3B (+Cl)"
    REPO_ID="Qwen/Qwen2.5-Omni-3B"
    BASE_MODEL_DIR="${MODEL_ROOT}/Qwen2.5-Omni-3B"
    CHECKPOINT="${MODEL_ROOT}/OEA-Qwen3B-Cl/step_40_inference_only.pt"
    CHECKPOINT_BYTES=59068647
    CHECKPOINT_SHA256="f084bf3c3ad645809e4c7e22cf148caef788cb96c019729576b881291268432a"
    CLOTHO_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_qwen3b_clotho_embeddings_seed42_20260719_181621"
    AUDIOCAPS_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_qwen3b_cl_audiocaps_main_20260802_190326/embeddings/audio_embeddings.npz"
    MECAT_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_qwen3b_cl_mecat_positive_uiq_public848_20260802_212321/audio_embeddings/audio_embeddings.npz"
    ;;
  oea_qwen7b)
    PAPER_MODEL="OEA-Qwen7B"
    REPO_ID="Qwen/Qwen2.5-Omni-7B"
    BASE_MODEL_DIR="${MODEL_ROOT}/Qwen2.5-Omni-7B"
    CHECKPOINT="${MODEL_ROOT}/oea-qwen7b-ac/step_300.pt"
    [[ -e "${CHECKPOINT}" ]] || CHECKPOINT="${MODEL_ROOT}/OEA-Qwen7B-AC/step_300.pt"
    CHECKPOINT_BYTES=17940602533
    CHECKPOINT_SHA256="cd751e3a71f0b47b9ecbbc0f5a11e4673097f0ebdbd80f610387f5778dd70a46"
    CLOTHO_AUDIO_SOURCE="${ROOT_DIR}/results/raw/official_source_oea_qwen7b_clotho_main_20260730_002924/embeddings/audio_embeddings.npz"
    AUDIOCAPS_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_qwen7b_audiocaps_main_20260802_125210/embeddings/audio_embeddings.npz"
    MECAT_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_qwen7b_mecat_positive_uiq_public848_20260802_133815/audio_embeddings/audio_embeddings.npz"
    ;;
  oea_qwen7b_cl)
    PAPER_MODEL="OEA-Qwen7B (+Cl)"
    REPO_ID="Qwen/Qwen2.5-Omni-7B"
    BASE_MODEL_DIR="${MODEL_ROOT}/Qwen2.5-Omni-7B"
    CHECKPOINT="${MODEL_ROOT}/oea-qwen7b-cl/step_330.pt"
    [[ -e "${CHECKPOINT}" ]] || CHECKPOINT="${MODEL_ROOT}/OEA-Qwen7B-Cl/step_330.pt"
    CHECKPOINT_BYTES=17940602661
    CHECKPOINT_SHA256="09ce41af7e7ac23106fa74d25e45b7229a046d687774cbd4ab2c00c05097d92e"
    CLOTHO_AUDIO_SOURCE="${ROOT_DIR}/results/raw/official_source_oea_qwen7b_cl_clotho_main_20260730_004037/embeddings/audio_embeddings.npz"
    AUDIOCAPS_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_qwen7b_cl_audiocaps_main_20260802_132013/embeddings/audio_embeddings.npz"
    MECAT_AUDIO_SOURCE="${ROOT_DIR}/results/raw/oea_qwen7b_cl_mecat_positive_uiq_public848_20260802_135201/audio_embeddings/audio_embeddings.npz"
    ;;
  *)
    echo "Unsupported model: ${VARIANT_ID}" >&2
    exit 2
    ;;
esac

CLOTHO_JSONL="${ROOT_DIR}/data/UIQ/clotho/clotho_evaluation_negative_queries.jsonl"
AUDIOCAPS_JSONL="${ROOT_DIR}/data/UIQ/audiocaps/audiocaps_test_negative_queries.jsonl"
MECAT_JSONL="${ROOT_DIR}/data/UIQ/mecat/mecat_negative_queries.jsonl"
CLOTHO_PAIRING="${PAIRING_ROOT}/clotho/pairing_metadata.jsonl"
AUDIOCAPS_PAIRING="${PAIRING_ROOT}/audiocaps/pairing_metadata.jsonl"
MECAT_PAIRING="${PAIRING_ROOT}/mecat/pairing_metadata.jsonl"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${VARIANT_ID}_negative_uiq_three_dataset_${STAMP}"
RESULT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}"
QUERY_ROOT="${RESULT_DIR}/query_embeddings"
METRICS_ROOT="${RESULT_DIR}/metrics"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"

[[ ! -e "${RESULT_DIR}" && ! -e "${LOG_DIR}" ]] || { echo "[ERROR] Refusing to reuse output." >&2; exit 2; }
mkdir -p "${QUERY_ROOT}/clotho" "${QUERY_ROOT}/audiocaps" "${QUERY_ROOT}/mecat" "${LOG_DIR}"
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
    (( remaining >= 0 )) || remaining=0
    echo "PROGRESS stage=${stage} stage_elapsed=$(format_duration "${elapsed}") stage_eta=$(format_duration "${remaining}")"
  done
  wait "${pid}"
  rc=$?
  echo "STAGE_END=${stage} RC=${rc}"
  return "${rc}"
}

echo "EXPERIMENT_NAME=${PAPER_MODEL} negative UIQ Tables 4 and 17 across Clotho, AudioCaps, MECAT"
echo "PAPER_EXPERIMENTS=EXP-16/17"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=${PAPER_MODEL}; released OEA checkpoint"
echo "PROTOCOL_STATUS=controlled inferred deterministic released-caption pairing"
echo "DATASETS=Clotho 1045/542; AudioCaps 975/630; MECAT public 848/409"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "OEA_SOURCE_FILES=AudioRetrieval/preprocessing/embeddings/oea.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py; AudioRetrieval/evaluation/negative_canonical.py"
echo "TOTAL_WORKLOAD=1,581 negative text embeddings with one model load; reuses all audio embeddings"
echo "ESTIMATED_TOTAL_TIME=3-12 minutes"
echo "DOWNLOADS_REQUIRED=no"
echo "OVERWRITE_DELETE_RISK=none; timestamped result and log directories"
echo "RESULT_DIRECTORY=${RESULT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

GIT_STATUS="$(git status --short | sed -E '/^\?\? scripts\/\.__dpc[[:xdigit:]]+$/d')"
[[ -z "${GIT_STATUS}" ]] || fail 3 "Git worktree is not clean"
for required in "${PAIRING_ROOT}" "${BASE_MODEL_DIR}" "${CHECKPOINT}" \
  "${CLOTHO_JSONL}" "${AUDIOCAPS_JSONL}" "${MECAT_JSONL}" \
  "${CLOTHO_PAIRING}" "${AUDIOCAPS_PAIRING}" "${MECAT_PAIRING}" \
  "${CLOTHO_AUDIO_SOURCE}" "${AUDIOCAPS_AUDIO_SOURCE}" "${MECAT_AUDIO_SOURCE}"; do
  [[ -e "${required}" ]] || fail 4 "Required resource is missing: ${required}"
done
[[ "$(stat -c %s "${CHECKPOINT}")" == "${CHECKPOINT_BYTES}" ]] || fail 4 "Checkpoint size mismatch"
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA256}" ]] || fail 4 "Checkpoint SHA256 mismatch"
[[ "$(sha256sum "${CLOTHO_PAIRING}" | awk '{print $1}')" == "c05101e76d6343a1446a0132bb16b041c03dcca7e9d96a62d61710177c9fb177" ]] || fail 4 "Clotho pairing SHA256 mismatch"
[[ "$(sha256sum "${AUDIOCAPS_PAIRING}" | awk '{print $1}')" == "8a773c5cd8519214baf1ebd715df89dd1b75ee685e563ba4bf9c0c8cf1e7effa" ]] || fail 4 "AudioCaps pairing SHA256 mismatch"
[[ "$(sha256sum "${MECAT_PAIRING}" | awk '{print $1}')" == "e7c7311281681190f544c5d9d0eed348fd3378bbc4ce9c31358096da36bab8aa" ]] || fail 4 "MECAT pairing SHA256 mismatch"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1" || fail 5 "Exactly one visible CUDA GPU is required"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv > "${RESULT_DIR}/gpu_info.txt" || fail 5 "nvidia-smi failed"

CLOTHO_AUDIO_NPZ="${CLOTHO_AUDIO_SOURCE}"
if [[ -d "${CLOTHO_AUDIO_SOURCE}" ]]; then
  CURRENT_STAGE="cpu_export_clotho_audio_npz"
  CLOTHO_EXPORT_DIR="${RESULT_DIR}/audio_inputs/clotho"
  run_stage "${CURRENT_STAGE}" 30 python scripts/export_candidate_audio_npz.py \
    --generation-dir "${CLOTHO_AUDIO_SOURCE}" --output-dir "${CLOTHO_EXPORT_DIR}" \
    --expected-candidates 1045 --dataset-name "Clotho v2.1 evaluation" \
    || fail $? "Clotho candidate NPZ export failed"
  CLOTHO_AUDIO_NPZ="${CLOTHO_EXPORT_DIR}/audio_embeddings.npz"
fi

CURRENT_STAGE="gpu_negative_query_embeddings"
run_stage "${CURRENT_STAGE}" 600 python scripts/generate_oea_negative_uiq_embeddings.py \
  --checkpoint "${CHECKPOINT}" --repo-id "${REPO_ID}" --local-path "${BASE_MODEL_DIR}" \
  --device cuda --batch-size-text 16 \
  --dataset-spec clotho "${CLOTHO_JSONL}" "${QUERY_ROOT}/clotho" 542 \
  --dataset-spec audiocaps "${AUDIOCAPS_JSONL}" "${QUERY_ROOT}/audiocaps" 630 \
  --dataset-spec mecat "${MECAT_JSONL}" "${QUERY_ROOT}/mecat" 409 \
  --summary "${RESULT_DIR}/query_embedding_summary.json" \
  || fail $? "${PAPER_MODEL} negative query embedding generation failed"

export CUDA_VISIBLE_DEVICES=""
for spec in \
  "clotho|Clotho v2.1 evaluation|${CLOTHO_AUDIO_NPZ}|${CLOTHO_PAIRING}|1045|542" \
  "audiocaps|AudioCaps v2 test|${AUDIOCAPS_AUDIO_SOURCE}|${AUDIOCAPS_PAIRING}|975|630" \
  "mecat|MECAT-Caption 00A/test public 848|${MECAT_AUDIO_SOURCE}|${MECAT_PAIRING}|848|409"; do
  IFS='|' read -r dataset label audio pairing candidates queries <<< "${spec}"
  CURRENT_STAGE="cpu_${dataset}_table17_metrics"
  run_stage "${CURRENT_STAGE}" 180 python scripts/evaluate_negative_uiq_npz.py \
    --audio-npz "${audio}" --query-npz "${QUERY_ROOT}/${dataset}/uiq_negative_embeddings.npz" \
    --pairing-jsonl "${pairing}" --output-dir "${METRICS_ROOT}/${dataset}" \
    --model "${PAPER_MODEL}" --dataset "${label}" \
    --expected-candidates "${candidates}" --expected-queries "${queries}" \
    || fail $? "${dataset} canonical negative metrics failed"
done

sha256sum \
  "${RESULT_DIR}/query_embedding_summary.json" \
  "${QUERY_ROOT}/clotho/uiq_negative_embeddings.npz" \
  "${QUERY_ROOT}/audiocaps/uiq_negative_embeddings.npz" \
  "${QUERY_ROOT}/mecat/uiq_negative_embeddings.npz" \
  "${METRICS_ROOT}/clotho/metrics.json" \
  "${METRICS_ROOT}/audiocaps/metrics.json" \
  "${METRICS_ROOT}/mecat/metrics.json" > "${RESULT_DIR}/artifact_sha256.txt" \
  || fail 8 "Artifact SHA256 manifest failed"

finish 0 complete none none
