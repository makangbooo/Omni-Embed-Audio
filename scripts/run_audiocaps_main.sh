#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <laion_clap|robust_clap|mga_clap|m2d_clap|oea_nemo3b|oea_nemo3b_cl|oea_qwen3b|oea_qwen3b_cl|oea_qwen7b|oea_qwen7b_cl>" >&2
  exit 2
fi

VARIANT_ID="$1"
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
DATASET_ROOT="${DATA_ROOT}/audiocaps_v2_d004db3"
MANIFEST="${DATASET_ROOT}/manifests/audiocaps_v2_test_audio_manifest.jsonl"
MANIFEST_SHA256="a341c9dfcb1b2cbe69675e8f1338edb502ce03f8984bf5882dc4d107b72144b4"
UIQ_ROOT="${ROOT_DIR}/data/UIQ/audiocaps"
BERT_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bert-base-uncased"
PYTHON_PREFIX=""
REQUIRED_RESOURCES=()
PRECHECK_IMPORT="import torch"

case "${VARIANT_ID}" in
  laion_clap)
    PAPER_MODEL="LAION-CLAP"
    BACKEND="laion_clap"
    EXPECTED_DIM=512
    BATCH_AUDIO=32
    BATCH_TEXT=128
    CHECKPOINT="${MODEL_ROOT}/laion-clap/630k-audioset-best.pt"
    CHECKPOINT_BYTES=1863587645
    CHECKPOINT_SHA256="8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
    ROBERTA_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/roberta-base"
    BART_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bart-base"
    PYTHON_PREFIX="${MODEL_ROOT}/python/laion-clap-1.1.6-overlay-v3"
    REQUIRED_RESOURCES=("${BERT_TOKENIZER}" "${ROBERTA_TOKENIZER}" "${BART_TOKENIZER}" "${PYTHON_PREFIX}")
    MODEL_ARGS=(
      --laion-ckpt "${CHECKPOINT}"
      --laion-bert-tokenizer "${BERT_TOKENIZER}"
      --laion-roberta-tokenizer "${ROBERTA_TOKENIZER}"
      --laion-bart-tokenizer "${BART_TOKENIZER}"
    )
    PRECHECK_IMPORT="import laion_clap, torch"
    OEA_SOURCE_FILES="AudioRetrieval/models/laion_clap_adapter.py; AudioRetrieval/preprocessing/embeddings/laion_clap.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py"
    ;;
  robust_clap)
    PAPER_MODEL="Robust-CLAP"
    BACKEND="robust_clap"
    EXPECTED_DIM=512
    BATCH_AUDIO=32
    BATCH_TEXT=128
    SOURCE_DIR="${MODEL_ROOT}/robust-clap/source"
    CHECKPOINT="${MODEL_ROOT}/laion-clap/630k-audioset-best.pt"
    CHECKPOINT_BYTES=1863587645
    CHECKPOINT_SHA256="8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
    ROBERTA_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/roberta-base"
    BART_TOKENIZER="${MODEL_ROOT}/laion-clap-tokenizers/bart-base"
    PYTHON_PREFIX="${MODEL_ROOT}/python/laion-clap-1.1.6-overlay-v3"
    BPE_VOCAB="${PYTHON_PREFIX}/laion_clap/clap_module/bpe_simple_vocab_16e6.txt.gz"
    REQUIRED_RESOURCES=("${SOURCE_DIR}" "${BERT_TOKENIZER}" "${ROBERTA_TOKENIZER}" "${BART_TOKENIZER}" "${PYTHON_PREFIX}" "${BPE_VOCAB}")
    MODEL_ARGS=(
      --robust-ckpt "${CHECKPOINT}"
      --robust-repo "${SOURCE_DIR}"
      --robust-bert-tokenizer "${BERT_TOKENIZER}"
      --robust-roberta-tokenizer "${ROBERTA_TOKENIZER}"
      --robust-bart-tokenizer "${BART_TOKENIZER}"
      --robust-bpe-vocab "${BPE_VOCAB}"
    )
    PRECHECK_IMPORT="import laion_clap, torch"
    OEA_SOURCE_FILES="AudioRetrieval/models/robust_clap_adapter.py; AudioRetrieval/preprocessing/embeddings/robust_clap.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py"
    ;;
  mga_clap)
    PAPER_MODEL="MGA-CLAP"
    BACKEND="mga_clap"
    EXPECTED_DIM=1024
    BATCH_AUDIO=32
    BATCH_TEXT=128
    SOURCE_DIR="${MODEL_ROOT}/mga-clap/source"
    CHECKPOINT="${MODEL_ROOT}/mga-clap/pretrained_models/models/model.pt"
    CHECKPOINT_BYTES=1711356348
    CHECKPOINT_SHA256="8703740b738e973a5b4d8a18a074ad56880e98f8ba21cd618d7d7ee5422d6e26"
    RUNTIME_OVERLAY="${MGA_RUNTIME_OVERLAY:-${MODEL_ROOT}/python/laion-clap-1.1.6-overlay-v3}"
    DEPENDENCY_OVERLAY="${MGA_DEPENDENCY_OVERLAY:-${MODEL_ROOT}/python/mga-clap-runtime-v1}"
    PYTHON_PREFIX="${DEPENDENCY_OVERLAY}:${RUNTIME_OVERLAY}"
    REQUIRED_RESOURCES=("${SOURCE_DIR}" "${BERT_TOKENIZER}" "${RUNTIME_OVERLAY}" "${DEPENDENCY_OVERLAY}")
    MODEL_ARGS=(
      --mga-repo "${SOURCE_DIR}"
      --mga-ckpt "${CHECKPOINT}"
      --mga-bert-tokenizer "${BERT_TOKENIZER}"
      --mga-checkpoint-sha256 "${CHECKPOINT_SHA256}"
    )
    PRECHECK_IMPORT="import ruamel.yaml, torch; from torchlibrosa.augmentation import SpecAugmentation"
    OEA_SOURCE_FILES="AudioRetrieval/models/mga_clap_adapter.py; AudioRetrieval/preprocessing/embeddings/mga_clap.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py"
    ;;
  m2d_clap)
    PAPER_MODEL="M2D-CLAP"
    BACKEND="m2d_clap"
    EXPECTED_DIM=768
    BATCH_AUDIO=8
    BATCH_TEXT=128
    CHECKPOINT="${MODEL_ROOT}/m2d-clap/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"
    CHECKPOINT_BYTES=1694677655
    CHECKPOINT_SHA256="238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1"
    REQUIRED_RESOURCES=("${BERT_TOKENIZER}")
    MODEL_ARGS=(--m2d-ckpt "${CHECKPOINT}" --m2d-bert-tokenizer "${BERT_TOKENIZER}")
    OEA_SOURCE_FILES="AudioRetrieval/models/m2d_clap_adapter.py; AudioRetrieval/preprocessing/embeddings/m2d_clap.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py"
    ;;
  oea_nemo3b)
    PAPER_MODEL="OEA-Nemo3B"
    BACKEND="oea"
    EXPECTED_DIM=512
    BATCH_AUDIO=16
    BATCH_TEXT=16
    BASE_MODEL_DIR="${MODEL_ROOT}/omni-embed-nemotron-3b"
    CHECKPOINT="${MODEL_ROOT}/OEA-Nemo3B-AC/step_400_best.pt"
    CHECKPOINT_BYTES=9466826153
    CHECKPOINT_SHA256="55579dfbd4f6621b5d842c5e731d6a1d37dbfd04b26b1c55bf8cdea980e67d25"
    REQUIRED_RESOURCES=("${BASE_MODEL_DIR}")
    MODEL_ARGS=(--checkpoint "${CHECKPOINT}" --repo-id nvidia/omni-embed-nemotron-3b --local-path "${BASE_MODEL_DIR}")
    OEA_SOURCE_FILES="AudioRetrieval/preprocessing/embeddings/oea.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py"
    ;;
  oea_nemo3b_cl)
    PAPER_MODEL="OEA-Nemo3B (+Cl)"
    BACKEND="oea"
    EXPECTED_DIM=512
    BATCH_AUDIO=16
    BATCH_TEXT=16
    BASE_MODEL_DIR="${MODEL_ROOT}/omni-embed-nemotron-3b"
    CHECKPOINT="${MODEL_ROOT}/OEA-Nemo3B-Cl/step_450_best_inference_only.pt"
    CHECKPOINT_BYTES=59072047
    CHECKPOINT_SHA256="2a5bee9039a28c0028cf205d1e2f4302fda540dd913f4cc1b08301edfc6680c4"
    REQUIRED_RESOURCES=("${BASE_MODEL_DIR}")
    MODEL_ARGS=(--checkpoint "${CHECKPOINT}" --repo-id nvidia/omni-embed-nemotron-3b --local-path "${BASE_MODEL_DIR}")
    OEA_SOURCE_FILES="AudioRetrieval/preprocessing/embeddings/oea.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py"
    ;;
  oea_qwen3b)
    PAPER_MODEL="OEA-Qwen3B"
    BACKEND="oea"
    EXPECTED_DIM=512
    BATCH_AUDIO=16
    BATCH_TEXT=16
    BASE_MODEL_DIR="${MODEL_ROOT}/Qwen2.5-Omni-3B"
    CHECKPOINT="${MODEL_ROOT}/OEA-Qwen3B-AC/step_350_inference_only.pt"
    CHECKPOINT_BYTES=59069203
    CHECKPOINT_SHA256="b1d0f559711b70f5a80dbdeb8cd46d80ed7b38b8e5524b9a871878d8bcd5f101"
    REQUIRED_RESOURCES=("${BASE_MODEL_DIR}")
    MODEL_ARGS=(--checkpoint "${CHECKPOINT}" --repo-id Qwen/Qwen2.5-Omni-3B --local-path "${BASE_MODEL_DIR}")
    OEA_SOURCE_FILES="AudioRetrieval/preprocessing/embeddings/oea.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py"
    ;;
  oea_qwen3b_cl)
    PAPER_MODEL="OEA-Qwen3B (+Cl)"
    BACKEND="oea"
    EXPECTED_DIM=512
    BATCH_AUDIO=16
    BATCH_TEXT=16
    BASE_MODEL_DIR="${MODEL_ROOT}/Qwen2.5-Omni-3B"
    CHECKPOINT="${MODEL_ROOT}/OEA-Qwen3B-Cl/step_40_inference_only.pt"
    CHECKPOINT_BYTES=59068647
    CHECKPOINT_SHA256="f084bf3c3ad645809e4c7e22cf148caef788cb96c019729576b881291268432a"
    REQUIRED_RESOURCES=("${BASE_MODEL_DIR}")
    MODEL_ARGS=(--checkpoint "${CHECKPOINT}" --repo-id Qwen/Qwen2.5-Omni-3B --local-path "${BASE_MODEL_DIR}")
    OEA_SOURCE_FILES="AudioRetrieval/preprocessing/embeddings/oea.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py"
    ;;
  oea_qwen7b)
    PAPER_MODEL="OEA-Qwen7B"
    BACKEND="oea"
    EXPECTED_DIM=512
    BATCH_AUDIO=16
    BATCH_TEXT=16
    BASE_MODEL_DIR="${MODEL_ROOT}/Qwen2.5-Omni-7B"
    CHECKPOINT="${MODEL_ROOT}/oea-qwen7b-ac/step_300.pt"
    if [[ ! -e "${CHECKPOINT}" ]]; then
      CHECKPOINT="${MODEL_ROOT}/OEA-Qwen7B-AC/step_300.pt"
    fi
    CHECKPOINT_BYTES=17940602533
    CHECKPOINT_SHA256="cd751e3a71f0b47b9ecbbc0f5a11e4673097f0ebdbd80f610387f5778dd70a46"
    REQUIRED_RESOURCES=("${BASE_MODEL_DIR}")
    MODEL_ARGS=(--checkpoint "${CHECKPOINT}" --repo-id Qwen/Qwen2.5-Omni-7B --local-path "${BASE_MODEL_DIR}")
    OEA_SOURCE_FILES="AudioRetrieval/preprocessing/embeddings/oea.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py"
    ;;
  oea_qwen7b_cl)
    PAPER_MODEL="OEA-Qwen7B (+Cl)"
    BACKEND="oea"
    EXPECTED_DIM=512
    BATCH_AUDIO=16
    BATCH_TEXT=16
    BASE_MODEL_DIR="${MODEL_ROOT}/Qwen2.5-Omni-7B"
    CHECKPOINT="${MODEL_ROOT}/oea-qwen7b-cl/step_330.pt"
    if [[ ! -e "${CHECKPOINT}" ]]; then
      CHECKPOINT="${MODEL_ROOT}/OEA-Qwen7B-Cl/step_330.pt"
    fi
    CHECKPOINT_BYTES=17940602661
    CHECKPOINT_SHA256="09ce41af7e7ac23106fa74d25e45b7229a046d687774cbd4ab2c00c05097d92e"
    REQUIRED_RESOURCES=("${BASE_MODEL_DIR}")
    MODEL_ARGS=(--checkpoint "${CHECKPOINT}" --repo-id Qwen/Qwen2.5-Omni-7B --local-path "${BASE_MODEL_DIR}")
    OEA_SOURCE_FILES="AudioRetrieval/preprocessing/embeddings/oea.py; AudioRetrieval/preprocessing/embeddings/uiq_text.py"
    ;;
  *)
    echo "[ERROR] Unsupported variant: ${VARIANT_ID}" >&2
    exit 2
    ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${VARIANT_ID}_audiocaps_main_${STAMP}"
RESULT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}"
EMBEDDING_DIR="${RESULT_DIR}/embeddings"
UIQ_DIR="${RESULT_DIR}/uiq_embeddings"
METRICS_DIR="${RESULT_DIR}/metrics"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"

if [[ -e "${RESULT_DIR}" || -e "${LOG_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse result or log directory." >&2
  exit 2
fi
mkdir -p "${EMBEDDING_DIR}" "${UIQ_DIR}" "${LOG_DIR}"
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

echo "EXPERIMENT_NAME=${PAPER_MODEL} AudioCaps main Tables 2/3 and positive UIQ Tables 12-15"
echo "PAPER_EXPERIMENTS=EXP-10 Table 2 T2A; EXP-11 Table 3 T2T; EXP-12-15 positive UIQ"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=${PAPER_MODEL}"
if [[ "${VARIANT_ID}" == "robust_clap" ]]; then
  echo "PROTOCOL_STATUS=controlled public-code reproduction; pinned upstream source with standard LAION checkpoint; strict paper checkpoint identity not claimed"
else
  echo "PROTOCOL_STATUS=controlled public-code reproduction"
fi
echo "DATASET=AudioCaps v2 test; 975 audio candidates; 4,875 captions; 3,900 released UIQ queries"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "OEA_SOURCE_FILES=${OEA_SOURCE_FILES}; AudioRetrieval/evaluation/metrics.py"
echo "TOTAL_WORKLOAD=975 audio + 4,875 caption + 3,900 UIQ embeddings + 8 retrieval protocols"
echo "ESTIMATED_TOTAL_TIME=10-30 minutes for CLAP variants; 20-60 minutes for OEA variants"
echo "DOWNLOADS_REQUIRED=no"
echo "OVERWRITE_DELETE_RISK=none; timestamped result and log directories"
echo "FAILURE_RESUME_BEHAVIOR=failed timestamped artifacts are retained; rerun creates a new directory"
echo "CHECKPOINT=${CHECKPOINT}"
echo "DATA_MANIFEST=${MANIFEST}"
echo "RESULT_DIRECTORY=${RESULT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

GIT_STATUS="$(git status --short | sed -E '/^\?\? scripts\/\.__dpc[[:xdigit:]]+$/d')"
if [[ -n "${GIT_STATUS}" ]]; then
  printf '%s\n' "${GIT_STATUS}" >&2
  fail 3 "Git worktree is not clean"
fi

UIQ_FILES=(
  "${UIQ_ROOT}/audiocaps_test_question_queries.jsonl"
  "${UIQ_ROOT}/audiocaps_test_imperative_queries.jsonl"
  "${UIQ_ROOT}/audiocaps_test_paraphrase_queries.jsonl"
  "${UIQ_ROOT}/audiocaps_test_tagging_queries.jsonl"
)
for required in "${MANIFEST}" "${CHECKPOINT}" "${UIQ_FILES[@]}" "${REQUIRED_RESOURCES[@]}"; do
  [[ -e "${required}" ]] || fail 4 "Required resource is missing: ${required}"
done
[[ "$(stat -c %s "${CHECKPOINT}")" == "${CHECKPOINT_BYTES}" ]] \
  || fail 4 "Checkpoint byte size mismatch"
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA256}" ]] \
  || fail 4 "Checkpoint SHA256 mismatch"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA256}" ]] \
  || fail 4 "AudioCaps manifest SHA256 mismatch"
if [[ "${VARIANT_ID}" == "robust_clap" ]]; then
  python scripts/validate_robust_clap_source.py \
    --source-dir "${SOURCE_DIR}" --output "${RESULT_DIR}/source_identity.json" \
    || fail 4 "Robust-CLAP source identity mismatch"
  [[ "$(stat -c %s "${BPE_VOCAB}")" == "1356917" ]] \
    || fail 4 "Robust-CLAP BPE vocabulary byte size mismatch"
  [[ "$(sha256sum "${BPE_VOCAB}" | awk '{print $1}')" == "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a" ]] \
    || fail 4 "Robust-CLAP BPE vocabulary SHA256 mismatch"
fi

printf '%s\n' "${GIT_COMMIT}" > "${RESULT_DIR}/git_commit.txt"
printf '%s\n' "${GIT_STATUS}" > "${RESULT_DIR}/git_status.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv \
  > "${RESULT_DIR}/gpu_info.txt" || fail 5 "nvidia-smi failed"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export PYTHONPATH="${PYTHON_PREFIX:+${PYTHON_PREFIX}:}${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1' \
  || fail 5 "Exactly one visible CUDA GPU is required"
python -c "${PRECHECK_IMPORT}" || fail 5 "Pinned runtime dependencies are unavailable"

CURRENT_STAGE="gpu_audio_caption_embeddings"
run_stage "${CURRENT_STAGE}" 1800 \
  python scripts/precompute_audiocaps_embeddings.py \
    --model "${BACKEND}" --manifest "${MANIFEST}" \
    --manifest-sha256 "${MANIFEST_SHA256}" --output-dir "${EMBEDDING_DIR}" \
    --expected-examples 975 --expected-captions 4875 --expected-dim "${EXPECTED_DIM}" \
    --device cuda --batch-size-audio "${BATCH_AUDIO}" --batch-size-text "${BATCH_TEXT}" \
    "${MODEL_ARGS[@]}" \
  || fail $? "${PAPER_MODEL} AudioCaps audio/caption embedding generation failed"

CURRENT_STAGE="gpu_uiq_embeddings"
UIQ_INPUT_ARGS=()
for uiq_file in "${UIQ_FILES[@]}"; do
  UIQ_INPUT_ARGS+=(--uiq-jsonl "${uiq_file}")
done
run_stage "${CURRENT_STAGE}" 900 \
  python -m AudioRetrieval preprocess uiq-embeddings \
    --model "${BACKEND}" --dataset "AudioCaps v2 test" \
    --device cuda --batch-size-text "${BATCH_TEXT}" \
    --output-dir "${UIQ_DIR}" "${UIQ_INPUT_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
  || fail $? "${PAPER_MODEL} AudioCaps UIQ embedding generation failed"

CURRENT_STAGE="cpu_table2_table3_table12_table15_metrics"
export CUDA_VISIBLE_DEVICES=""
run_stage "${CURRENT_STAGE}" 300 \
  python scripts/evaluate_audiocaps_main.py \
    --embedding-dir "${EMBEDDING_DIR}" --uiq-dir "${UIQ_DIR}" \
    --manifest "${MANIFEST}" --manifest-sha256 "${MANIFEST_SHA256}" \
    --output-dir "${METRICS_DIR}" --model "${PAPER_MODEL}" \
    --expected-candidates 975 --captions-per-audio 5 --seed 0 \
  || fail $? "${PAPER_MODEL} AudioCaps metric finalization failed"

sha256sum "${EMBEDDING_DIR}/generation_metrics.json" \
  "${EMBEDDING_DIR}/audio_embeddings.npz" \
  "${EMBEDDING_DIR}/caption_embeddings.npz" \
  "${UIQ_DIR}"/uiq_*_embeddings.npz \
  "${METRICS_DIR}/suite_metrics.json" "${LOG_FILE}" \
  > "${RESULT_DIR}/artifact_sha256.txt" \
  || fail $? "Final artifact hashing failed"

finish 0 complete none none
