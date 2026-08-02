#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <vanilla_nemotron_3b|vanilla_qwen2_5_omni_3b|vanilla_qwen2_5_omni_7b>" >&2
  exit 2
fi

BACKBONE_ID="$1"
case "${BACKBONE_ID}" in
  vanilla_nemotron_3b)
    PAPER_MODEL="Nemotron-3B"
    EVIDENCE="results/audits/vanilla_nemotron_3b_clotho_main_eval_20260730.json"
    ;;
  vanilla_qwen2_5_omni_3b)
    PAPER_MODEL="Qwen2.5-Omni-3B"
    EVIDENCE="results/audits/vanilla_qwen2_5_omni_3b_clotho_main_eval_20260730.json"
    ;;
  vanilla_qwen2_5_omni_7b)
    PAPER_MODEL="Qwen2.5-Omni-7B"
    EVIDENCE="results/audits/vanilla_qwen2_5_omni_7b_clotho_main_eval_20260730.json"
    ;;
  *)
    echo "[ERROR] Unsupported vanilla backbone: ${BACKBONE_ID}" >&2
    exit 2
    ;;
esac

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MANIFEST="${DATA_ROOT}/audiocaps_v2_d004db3/manifests/audiocaps_v2_test_audio_manifest.jsonl"
MANIFEST_SHA256="a341c9dfcb1b2cbe69675e8f1338edb502ce03f8984bf5882dc4d107b72144b4"
MODEL_LOCK="${ROOT_DIR}/results/model_locks/${BACKBONE_ID}.json"
PROTOCOL_CONFIG="${ROOT_DIR}/configs/eval/${BACKBONE_ID}_audiocaps_embeddings.json"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${BACKBONE_ID}_audiocaps_table2_table3_${STAMP}"
RESULT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}"
GENERATION_DIR="${RESULT_DIR}/${BACKBONE_ID}_audiocaps_embeddings_seed42_${STAMP}"
NPZ_DIR="${RESULT_DIR}/embeddings"
METRICS_DIR="${RESULT_DIR}/metrics"
ATTEMPT_DIR="${RESULT_DIR}/attempt"
RESOLVED_CONFIG="${RESULT_DIR}/resolved_embedding_config.json"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"

if [[ -e "${RESULT_DIR}" || -e "${LOG_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse result or log directory." >&2
  exit 2
fi
mkdir -p "${RESULT_DIR}" "${ATTEMPT_DIR}" "${LOG_DIR}"
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

echo "EXPERIMENT_NAME=${PAPER_MODEL} vanilla AudioCaps Tables 2 and 3"
echo "PAPER_EXPERIMENTS=EXP-10 Table 2 T2A; EXP-11 Table 3 T2T"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=${PAPER_MODEL}; base-only; no OEA checkpoint, LoRA, or projection head"
echo "PROTOCOL_STATUS=controlled public-code reproduction"
echo "DATASET=AudioCaps v2 test; 975 audio candidates; 4,875 captions"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "OEA_SOURCE_FILES=AudioRetrieval/models/omni_embed_adapter.py; scripts/generate_vanilla_backbone_embeddings.py; AudioRetrieval/evaluation/metrics.py"
echo "TOTAL_WORKLOAD=975 audio + 4,875 caption embeddings + 4 retrieval protocols"
echo "ESTIMATED_TOTAL_TIME=20-90 minutes depending on backbone and GPU"
echo "DOWNLOADS_REQUIRED=no"
echo "OVERWRITE_DELETE_RISK=none; timestamped result and log directories"
echo "MODEL_CHECKPOINT_LOADED=no"
echo "RESULT_DIRECTORY=${RESULT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

GIT_STATUS="$(git status --short | sed -E '/^\?\? scripts\/\.__dpc[[:xdigit:]]+$/d')"
[[ -z "${GIT_STATUS}" ]] || fail 3 "Git worktree is not clean"
for required in "${MANIFEST}" "${MODEL_LOCK}" "${PROTOCOL_CONFIG}" "${ROOT_DIR}/${EVIDENCE}"; do
  [[ -e "${required}" ]] || fail 4 "Required resource is missing: ${required}"
done
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA256}" ]] \
  || fail 4 "AudioCaps manifest SHA256 mismatch"
python - "${ROOT_DIR}/${EVIDENCE}" <<'PY' || fail 4 "Completed Clotho model-load evidence is invalid"
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value.get("status") == "complete"
PY

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

CURRENT_STAGE="config_resolution"
run_stage "${CURRENT_STAGE}" 30 \
  python scripts/build_vanilla_backbone_eval_config.py \
    --protocol-config "${PROTOCOL_CONFIG}" --model-lock "${MODEL_LOCK}" \
    --output "${RESOLVED_CONFIG}" \
  || fail $? "Vanilla AudioCaps config resolution failed"

CURRENT_STAGE="gpu_audio_caption_embeddings"
run_stage "${CURRENT_STAGE}" 5400 \
  python scripts/generate_vanilla_backbone_embeddings.py \
    --config "${RESOLVED_CONFIG}" --model-root "${MODEL_ROOT}" \
    --manifest "${MANIFEST}" --output-dir "${GENERATION_DIR}" \
    --attempt-dir "${ATTEMPT_DIR}" \
  || fail $? "Vanilla AudioCaps embedding generation failed"

CURRENT_STAGE="canonical_npz_export"
run_stage "${CURRENT_STAGE}" 300 \
  python scripts/export_vanilla_audiocaps_npz.py \
    --generation-dir "${GENERATION_DIR}" --output-dir "${NPZ_DIR}" \
  || fail $? "Vanilla AudioCaps canonical export failed"

CURRENT_STAGE="cpu_table2_table3_metrics"
export CUDA_VISIBLE_DEVICES=""
run_stage "${CURRENT_STAGE}" 300 \
  python scripts/evaluate_audiocaps_main.py \
    --embedding-dir "${NPZ_DIR}" --uiq-dir "${RESULT_DIR}/unused_uiq" \
    --manifest "${MANIFEST}" --manifest-sha256 "${MANIFEST_SHA256}" \
    --output-dir "${METRICS_DIR}" --model "${PAPER_MODEL}" \
    --expected-candidates 975 --captions-per-audio 5 --seed 0 --skip-uiq \
  || fail $? "Vanilla AudioCaps Table 2/3 finalization failed"

sha256sum "${RESOLVED_CONFIG}" "${GENERATION_DIR}/generation_metrics.json" \
  "${NPZ_DIR}/conversion_metrics.json" "${NPZ_DIR}/audio_embeddings.npz" \
  "${NPZ_DIR}/caption_embeddings.npz" "${METRICS_DIR}/suite_metrics.json" \
  "${LOG_FILE}" > "${RESULT_DIR}/artifact_sha256.txt" \
  || fail $? "Final artifact hashing failed"

finish 0 complete none none
