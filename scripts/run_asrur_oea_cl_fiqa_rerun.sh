#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ( "$1" != "--dry-run" && "$1" != "--execute" ) ]]; then
  echo "Usage: bash scripts/run_asrur_oea_cl_fiqa_rerun.sh <--dry-run|--execute>" >&2
  exit 2
fi

MODE="$1"
if [[ "${MODE}" == "--execute" && -z "${TMUX:-}" ]]; then
  echo "[ERROR] GPU rerun must run inside tmux." >&2
  echo "[INFO] Start one with: tmux new -s asrur_oea_cl_fiqa_rerun" >&2
  exit 3
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MODELS_ROOT="${MODELS_ROOT:-/home/jg525/models}"
OEA_MODEL_ROOT="${OEA_MODEL_ROOT:-${MODELS_ROOT}/oea}"
MAIN_CONFIG="${ROOT_DIR}/configs/asr_uncertainty_reranking/main_experiment.json"
MODEL_RESOURCE_MANIFEST="${ROOT_DIR}/configs/asr_uncertainty_reranking/resources/models.json"
OEA_PROTOCOL="${ROOT_DIR}/configs/eval/nemo3b_cl_clotho_embeddings.json"
OEA_LOCK="${ROOT_DIR}/results/model_locks/oea_nemo3b_cl.json"
CORPUS="${DATA_ROOT}/fiqa_mteb/corpus.jsonl"
TEST_QRELS="${DATA_ROOT}/fiqa_mteb/qrels/test.jsonl"
SQuTR_MANIFEST="${DATA_ROOT}/squtr/manifests/squtr_en_fiqa_nq_audio_query_manifest.jsonl"
REFERENCE_CACHE_ROOT="${ASRUR_OEA_CL_REFERENCE_CACHE_ROOT:-/home/jg525/experiment_cache/asr_uncertainty/fiqa_phase2_d9baf226c075}"
REFERENCE_RESULT_ROOT="${ASRUR_OEA_CL_REFERENCE_RESULT_ROOT:-${ROOT_DIR}/results/raw/asrur_phase2_fiqa_d9baf226c075}"
GIT_COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="${GIT_COMMIT:0:12}"
CACHE_ROOT="${ASRUR_OEA_CL_RERUN_CACHE_ROOT:-/home/jg525/experiment_cache/asr_uncertainty/oea_cl_fiqa_rerun_${COMMIT_SHORT}}"
RESULT_ROOT="${ASRUR_OEA_CL_RERUN_RESULT_ROOT:-${ROOT_DIR}/results/raw/asrur_oea_cl_fiqa_rerun_${COMMIT_SHORT}}"
RESOLVED_CONFIG="${CACHE_ROOT}/resolved_configs/oea_nemo3b_cl.json"
RUN_ID="asrur_oea_cl_fiqa_rerun_${MODE#--}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
AUDIT_OUTPUT="${RUN_DIR}/oea_cl_fiqa_rerun_audit.json"
CONDITIONS=(clean snr_20 snr_10 snr_0)
TEXT_BATCH_SIZE="${ASRUR_OEA_TEXT_BATCH_SIZE:-4}"
AUDIO_BATCH_SIZE="${ASRUR_OEA_AUDIO_BATCH_SIZE:-1}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[ERROR] Formal rerun requires a clean Git worktree." >&2
  git status --short --untracked-files=all >&2
  exit 4
fi
for path in \
  "${MAIN_CONFIG}" \
  "${MODEL_RESOURCE_MANIFEST}" \
  "${OEA_PROTOCOL}" \
  "${OEA_LOCK}" \
  "${CORPUS}" \
  "${TEST_QRELS}" \
  "${SQuTR_MANIFEST}"
do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing prerequisite: ${path}" >&2
    exit 5
  fi
done
if [[ "$(readlink -m "${CACHE_ROOT}")" == "$(readlink -m "${REFERENCE_CACHE_ROOT}")" ]]; then
  echo "[ERROR] Fresh and reference cache roots must differ." >&2
  exit 6
fi
if [[ -L "${CACHE_ROOT}" ]]; then
  echo "[ERROR] Fresh cache root must not be a symlink: ${CACHE_ROOT}" >&2
  exit 6
fi
if [[ -d "${CACHE_ROOT}" ]] \
  && find "${CACHE_ROOT}" -type l -print -quit | grep -q .; then
  echo "[ERROR] Fresh cache root contains a symlink; old-cache reuse is prohibited." >&2
  exit 6
fi

mkdir -p "${RUN_DIR}/steps" "${CACHE_ROOT}/resolved_configs" "${RESULT_ROOT}"

record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
}
trap record_wrapper_exit EXIT
exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

run_step() {
  local name=$1
  shift
  local step_dir="${RUN_DIR}/steps/${name}"
  local started finished elapsed exit_code
  mkdir -p "${step_dir}"
  {
    printf '%q ' "$@"
    printf '\n'
  } > "${step_dir}/command.sh"
  started="$(date +%s)"
  date -Is > "${step_dir}/started_at.txt"
  echo "[STEP] ${name}"
  set +e
  "$@" > >(tee "${step_dir}/stdout.log") \
    2> >(tee "${step_dir}/stderr.log" >&2)
  exit_code=$?
  set -e
  finished="$(date +%s)"
  elapsed=$((finished - started))
  date -Is > "${step_dir}/finished_at.txt"
  printf '%s\n' "${elapsed}" > "${step_dir}/elapsed_seconds.txt"
  printf '%s\n' "${exit_code}" > "${step_dir}/exit_code.txt"
  printf '[TIMING] step=%s elapsed_seconds=%s exit_code=%s\n' \
    "${name}" "${elapsed}" "${exit_code}"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "[ERROR] Step ${name} exited ${exit_code}; fresh partial caches were preserved." >&2
    return "${exit_code}"
  fi
}

run_cache_step() {
  local name=$1
  local output_dir=$2
  shift 2
  if [[ "${MODE}" == "--execute" && -f "${output_dir}/cache_manifest.json" ]]; then
    echo "[STEP] ${name}"
    echo "[INFO] Resuming verified complete cache created inside the fresh rerun root"
    python - "${output_dir}/cache_manifest.json" <<'PY'
import sys
from AudioRetrieval.asr_uncertainty_reranking.cache_manifest import (
    load_cache_manifest,
    verify_file_records,
)
manifest = load_cache_manifest(sys.argv[1])
mismatches = verify_file_records(manifest["outputs"])
if mismatches:
    raise RuntimeError("\n".join(mismatches))
print(f"[INFO] verified_outputs={len(manifest['outputs'])}")
PY
    return 0
  fi
  run_step "${name}" "$@"
}

{
  echo "git_commit=${GIT_COMMIT}"
  echo "mode=${MODE}"
  echo "checkpoint=JudeJiwoo/OEA-Nemo3B-Cl"
  echo "cache_root=${CACHE_ROOT}"
  echo "result_root=${RESULT_ROOT}"
  echo "reference_cache_root=${REFERENCE_CACHE_ROOT}"
  echo "reference_result_root=${REFERENCE_RESULT_ROOT}"
  echo "old_oea_embedding_reuse=0"
  echo "checkpoint_selection=0"
  echo "training=0"
  echo "network=0"
  echo "audio_protocol=audio_only_no_text_prefix"
  echo "text_protocol=query_prefix"
  echo "conditions=clean,snr_20,snr_10,snr_0"
} | tee "${RUN_DIR}/run_identity.txt"
git status --short --untracked-files=all > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
df -hT "${DATA_ROOT}" "${MODELS_ROOT}" "$(dirname "${CACHE_ROOT}")" \
  > "${RUN_DIR}/disk_before.txt" 2>&1 || true

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

DRY_ARGUMENTS=()
if [[ "${MODE}" == "--dry-run" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  DRY_ARGUMENTS=(--dry-run)
  echo "[INFO] CPU-only dry-run; no model is loaded."
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  run_step gpu_preflight \
    python scripts/validate_single_bf16_gpu.py \
      --output "${RUN_DIR}/gpu_preflight.json"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  if [[ "${GPU_NAME}" != *"RTX 4090"* ]]; then
    echo "[ERROR] Approved hardware is one RTX 4090; observed: ${GPU_NAME}" >&2
    exit 7
  fi
  run_step d2_d4_strict_audit \
    python scripts/verify_asrur_model_assets.py \
      --manifest "${MODEL_RESOURCE_MANIFEST}" \
      --model-root "${MODELS_ROOT}" \
      --output "${RUN_DIR}/d2_d4_model_audit.json"
fi

run_step resolve_oea \
  python scripts/build_official_oea_eval_config.py \
    --protocol-config "${OEA_PROTOCOL}" \
    --model-lock "${OEA_LOCK}" \
    --output "${RESOLVED_CONFIG}"

for condition in "${CONDITIONS[@]}"; do
  content="audio_only"
  if [[ "${condition}" == "clean" ]]; then
    content="both"
  fi
  run_cache_step "oea_${condition}" "${CACHE_ROOT}/oea/${condition}" \
    python scripts/generate_asrur_omni_caches.py \
      --mode oea \
      --main-config "${MAIN_CONFIG}" \
      --resolved-model-config "${RESOLVED_CONFIG}" \
      --model-root "${OEA_MODEL_ROOT}" \
      --corpus "${CORPUS}" \
      --audio-manifest "${SQuTR_MANIFEST}" \
      --subset fiqa \
      --conditions "${condition}" \
      --text-batch-size "${TEXT_BATCH_SIZE}" \
      --audio-batch-size "${AUDIO_BATCH_SIZE}" \
      --content "${content}" \
      --output-dir "${CACHE_ROOT}/oea/${condition}" \
      "${DRY_ARGUMENTS[@]}"
done

if [[ "${MODE}" == "--dry-run" ]]; then
  echo "[INFO] OEA-Nemo3B-Cl FiQA rerun dry-run completed."
  echo "[INFO] Run directory: ${RUN_DIR}"
  exit 0
fi

mkdir -p "${CACHE_ROOT}/rankings/oea"
for condition in "${CONDITIONS[@]}"; do
  run_step "dense_oea_${condition}" \
    python scripts/generate_asrur_frozen_caches.py dense \
      --config "${MAIN_CONFIG}" \
      --query-embeddings "${CACHE_ROOT}/oea/${condition}/audio_embeddings.npy" \
      --query-ids "${CACHE_ROOT}/oea/${condition}/audio_ids.jsonl" \
      --query-id-field id \
      --document-embeddings "${CACHE_ROOT}/oea/clean/document_embeddings.npy" \
      --document-ids "${CACHE_ROOT}/oea/clean/document_ids.jsonl" \
      --document-id-field id \
      --output "${CACHE_ROOT}/rankings/oea/${condition}.jsonl"
  if [[ -f "${RESULT_ROOT}/${condition}/metrics.json" ]]; then
    echo "[STEP] evaluate_${condition}"
    echo "[INFO] Reusing existing metrics from this fresh rerun root"
  else
    run_step "evaluate_${condition}" \
      python scripts/evaluate_asrur_dense_rankings.py \
        --ranking "B3_oea=${CACHE_ROOT}/rankings/oea/${condition}.jsonl" \
        --qrels "${TEST_QRELS}" \
        --output-dir "${RESULT_ROOT}/${condition}"
  fi
done

if [[ ! -d "${REFERENCE_CACHE_ROOT}" || ! -d "${REFERENCE_RESULT_ROOT}" ]]; then
  echo "[ERROR] Reference Phase-2 roots are absent; cannot finish independent audit." >&2
  exit 8
fi
run_step audit_fresh_against_reference \
  python scripts/audit_asrur_oea_cl_fiqa_rerun.py \
    --fresh-cache-root "${CACHE_ROOT}" \
    --reference-cache-root "${REFERENCE_CACHE_ROOT}" \
    --fresh-result-root "${RESULT_ROOT}" \
    --reference-result-root "${REFERENCE_RESULT_ROOT}" \
    --output "${AUDIT_OUTPUT}"

python - "${RUN_DIR}" "${CACHE_ROOT}" "${RESULT_ROOT}" "${AUDIT_OUTPUT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir, cache_root, result_root, audit = map(Path, sys.argv[1:])
payload = {
    "schema_version": 1,
    "status": "complete",
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "cache_root": str(cache_root.resolve()),
    "result_root": str(result_root.resolve()),
    "audit": str(audit.resolve()),
    "condition_metrics": {
        condition: str((result_root / condition / "metrics.json").resolve())
        for condition in ("clean", "snr_20", "snr_10", "snr_0")
    },
}
(run_dir / "completion_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

df -hT "$(dirname "${CACHE_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
nvidia-smi > "${RUN_DIR}/gpu_final.txt"
echo "[INFO] Independent OEA-Nemo3B-Cl FiQA rerun completed."
echo "[INFO] Cache root: ${CACHE_ROOT}"
echo "[INFO] Result root: ${RESULT_ROOT}"
echo "[INFO] Audit: ${AUDIT_OUTPUT}"
echo "[INFO] Run directory: ${RUN_DIR}"
