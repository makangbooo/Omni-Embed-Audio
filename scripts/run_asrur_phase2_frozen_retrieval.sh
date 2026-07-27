#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ( "$1" != "--dry-run" && "$1" != "--execute" ) ]]; then
  echo "Usage: $0 <--dry-run|--execute>" >&2
  echo "--execute requires a separate user GPU approval." >&2
  exit 2
fi

MODE="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MODELS_ROOT="${MODELS_ROOT:-/home/jg525/models}"
OEA_MODEL_ROOT="${OEA_MODEL_ROOT:-${MODELS_ROOT}/oea}"
MAIN_CONFIG="${ROOT_DIR}/configs/asr_uncertainty_reranking/main_experiment.json"
MODEL_RESOURCE_MANIFEST="${ROOT_DIR}/configs/asr_uncertainty_reranking/resources/models.json"
FIQA_ROOT="${DATA_ROOT}/fiqa_mteb"
CORPUS="${FIQA_ROOT}/corpus.jsonl"
QUERIES="${FIQA_ROOT}/queries.jsonl"
DEV_QRELS="${FIQA_ROOT}/qrels/dev.jsonl"
TEST_QRELS="${FIQA_ROOT}/qrels/test.jsonl"
SQuTR_MANIFEST="${DATA_ROOT}/squtr/manifests/squtr_en_fiqa_nq_audio_query_manifest.jsonl"
OEA_PROTOCOL="${ROOT_DIR}/configs/eval/nemo3b_cl_clotho_embeddings.json"
OEA_LOCK="${ROOT_DIR}/results/model_locks/oea_nemo3b_cl.json"
VANILLA_PROTOCOL="${ROOT_DIR}/configs/eval/vanilla_nemotron_3b_clotho_embeddings.json"
VANILLA_LOCK="${ROOT_DIR}/results/model_locks/vanilla_nemotron_3b.json"
GIT_COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="${GIT_COMMIT:0:12}"
CACHE_ROOT="${PHASE2_CACHE_ROOT:-/home/jg525/experiment_cache/asr_uncertainty/fiqa_phase2_${COMMIT_SHORT}}"
REUSE_CACHE_ROOT="${PHASE2_REUSE_CACHE_ROOT:-}"
REQUIRED_REUSE_COUNT="${PHASE2_REQUIRED_REUSE_COUNT:-0}"
RESULT_ROOT="${PHASE2_RESULT_ROOT:-${ROOT_DIR}/results/raw/asrur_phase2_fiqa_${COMMIT_SHORT}}"
RUN_ID="asrur_phase2_frozen_retrieval_${MODE#--}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
RESOLVED_ROOT="${CACHE_ROOT}/resolved_configs"
OEA_RESOLVED="${RESOLVED_ROOT}/oea_nemo3b_cl.json"
VANILLA_RESOLVED="${RESOLVED_ROOT}/vanilla_nemotron_3b.json"
CONDITIONS=(clean snr_20 snr_10 snr_0)
TEXT_BATCH_SIZE="${ASRUR_OEA_TEXT_BATCH_SIZE:-4}"
AUDIO_BATCH_SIZE="${ASRUR_OEA_AUDIO_BATCH_SIZE:-1}"
BGE_BATCH_SIZE="${ASRUR_BGE_BATCH_SIZE:-64}"
WHISPER_MAX_RECORD_ATTEMPTS="${ASRUR_WHISPER_MAX_RECORD_ATTEMPTS:-1}"
WHISPER_RESUME_SOURCE_ROOT="${PHASE2_WHISPER_RESUME_SOURCE_ROOT:-}"
WHISPER_RESUME_SOURCE_GIT_COMMIT="${PHASE2_WHISPER_RESUME_SOURCE_GIT_COMMIT:-}"

if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse run directory: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}/steps"

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
  mkdir -p "${step_dir}"
  {
    printf '%q ' "$@"
    printf '\n'
  } > "${step_dir}/command.sh"
  echo "[STEP] ${name}"
  set +e
  "$@" > >(tee "${step_dir}/stdout.log") \
    2> >(tee "${step_dir}/stderr.log" >&2)
  local exit_code=$?
  set -e
  printf '%s\n' "${exit_code}" > "${step_dir}/exit_code.txt"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "[ERROR] Step ${name} exited ${exit_code}; caches were preserved." >&2
    return "${exit_code}"
  fi
}

record_reused_step() {
  local name=$1
  local source=$2
  local destination=$3
  local step_dir="${RUN_DIR}/steps/${name}"
  mkdir -p "${step_dir}"
  printf 'reuse %q %q\n' "${source}" "${destination}" > "${step_dir}/command.sh"
  {
    echo "[INFO] Reused immutable complete cache"
    echo "source=${source}"
    echo "destination=${destination}"
    sha256sum "${source}/cache_manifest.json"
  } | tee "${step_dir}/stdout.log"
  : > "${step_dir}/stderr.log"
  printf '0\n' > "${step_dir}/exit_code.txt"
}

reuse_complete_cache_dir() {
  local source=$1
  local destination=$2
  local destination_already_linked=0
  if [[ "${MODE}" != "--execute" || -z "${REUSE_CACHE_ROOT}" ]]; then
    return 0
  fi
  if [[ ! -f "${source}/cache_manifest.json" ]]; then
    return 0
  fi
  if [[ -L "${destination}" ]]; then
    if [[ "$(readlink -f "${destination}")" != "$(readlink -f "${source}")" ]]; then
      echo "[ERROR] Existing reuse link targets a different cache: ${destination}" >&2
      return 7
    fi
    destination_already_linked=1
  elif [[ -e "${destination}" ]]; then
    return 0
  fi
  python - "${source}/cache_manifest.json" <<'PY'
import sys

from AudioRetrieval.asr_uncertainty_reranking.cache_manifest import (
    load_cache_manifest,
    verify_file_records,
)

manifest_path = sys.argv[1]
manifest = load_cache_manifest(manifest_path)
mismatches = verify_file_records(manifest["outputs"])
if mismatches:
    raise RuntimeError(
        "refusing cross-commit cache reuse:\n- " + "\n- ".join(mismatches)
    )
print(
    f"[INFO] Verified {len(manifest['outputs'])} immutable outputs "
    f"before cross-commit reuse: {manifest_path}"
)
PY
  if [[ "${destination_already_linked}" -eq 0 ]]; then
    mkdir -p "$(dirname "${destination}")"
    ln -s "${source}" "${destination}"
  fi
  printf '{"source":"%s","destination":"%s","manifest_sha256":"%s"}\n' \
    "${source}" \
    "${destination}" \
    "$(sha256sum "${source}/cache_manifest.json" | awk '{print $1}')" \
    >> "${RUN_DIR}/reused_cache_dirs.jsonl"
}

run_cache_step() {
  local name=$1
  local output_dir=$2
  shift 2
  if [[ "${MODE}" == "--execute" && -f "${output_dir}/cache_manifest.json" ]]; then
    record_reused_step "${name}" "$(readlink -f "${output_dir}")" "${output_dir}"
    return 0
  fi
  run_step "${name}" "$@"
}

if [[ ! "${REQUIRED_REUSE_COUNT}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] PHASE2_REQUIRED_REUSE_COUNT must be a non-negative integer." >&2
  exit 3
fi
if [[ ! "${WHISPER_MAX_RECORD_ATTEMPTS}" =~ ^[1-3]$ ]]; then
  echo "[ERROR] ASRUR_WHISPER_MAX_RECORD_ATTEMPTS must be 1, 2, or 3." >&2
  exit 3
fi
if [[ -n "${WHISPER_RESUME_SOURCE_ROOT}" || -n "${WHISPER_RESUME_SOURCE_GIT_COMMIT}" ]]; then
  if [[ -z "${WHISPER_RESUME_SOURCE_ROOT}" || -z "${WHISPER_RESUME_SOURCE_GIT_COMMIT}" ]]; then
    echo "[ERROR] Whisper resume source root and Git commit must be supplied together." >&2
    exit 3
  fi
  if [[ "${WHISPER_RESUME_SOURCE_ROOT}" != /* || ! -d "${WHISPER_RESUME_SOURCE_ROOT}" ]]; then
    echo "[ERROR] PHASE2_WHISPER_RESUME_SOURCE_ROOT must be an existing absolute directory." >&2
    exit 3
  fi
  if [[ ! "${WHISPER_RESUME_SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "[ERROR] PHASE2_WHISPER_RESUME_SOURCE_GIT_COMMIT must be a full lowercase commit." >&2
    exit 3
  fi
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[ERROR] Formal Phase-2 runner requires a clean Git worktree." >&2
  git status --short --untracked-files=all >&2
  exit 3
fi
for tracked in \
  "${MAIN_CONFIG}" \
  "${MODEL_RESOURCE_MANIFEST}" \
  "${OEA_PROTOCOL}" \
  "${OEA_LOCK}" \
  "${VANILLA_PROTOCOL}" \
  "${VANILLA_LOCK}"
do
  if [[ ! -f "${tracked}" ]]; then
    echo "[ERROR] Missing tracked prerequisite: ${tracked}" >&2
    exit 4
  fi
  git ls-files --error-unmatch -- "${tracked#${ROOT_DIR}/}" >/dev/null
done
for input in "${CORPUS}" "${QUERIES}" "${DEV_QRELS}" "${TEST_QRELS}" "${SQuTR_MANIFEST}"; do
  if [[ ! -f "${input}" ]]; then
    echo "[ERROR] Missing fixed data input: ${input}" >&2
    exit 4
  fi
done

{
  printf 'git_commit=%s\n' "${GIT_COMMIT}"
  printf 'git_status_short=\n'
  printf 'mode=%s\n' "${MODE}"
  printf 'cache_root=%s\n' "${CACHE_ROOT}"
  printf 'reuse_cache_root=%s\n' "${REUSE_CACHE_ROOT}"
  printf 'required_reuse_count=%s\n' "${REQUIRED_REUSE_COUNT}"
  printf 'result_root=%s\n' "${RESULT_ROOT}"
  printf 'text_batch_size=%s\n' "${TEXT_BATCH_SIZE}"
  printf 'audio_batch_size=%s\n' "${AUDIO_BATCH_SIZE}"
  printf 'bge_batch_size=%s\n' "${BGE_BATCH_SIZE}"
  printf 'whisper_max_record_attempts=%s\n' "${WHISPER_MAX_RECORD_ATTEMPTS}"
  printf 'whisper_resume_source_root=%s\n' "${WHISPER_RESUME_SOURCE_ROOT}"
  printf 'whisper_resume_source_git_commit=%s\n' "${WHISPER_RESUME_SOURCE_GIT_COMMIT}"
} > "${RUN_DIR}/run_identity.txt"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
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
    echo "[ERROR] Approved Phase-2 hardware is one RTX 4090; observed: ${GPU_NAME}" >&2
    exit 5
  fi
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
    --format=csv > "${RUN_DIR}/gpu_info.txt"
  run_step d2_d4_strict_audit \
    python scripts/verify_asrur_model_assets.py \
      --manifest "${MODEL_RESOURCE_MANIFEST}" \
      --model-root "${MODELS_ROOT}" \
      --output "${RUN_DIR}/d2_d4_model_audit.json"
fi

mkdir -p "${RESOLVED_ROOT}" "${RESULT_ROOT}"
if [[ -n "${REUSE_CACHE_ROOT}" ]]; then
  if [[ "${REUSE_CACHE_ROOT}" != /* || ! -d "${REUSE_CACHE_ROOT}" ]]; then
    echo "[ERROR] PHASE2_REUSE_CACHE_ROOT must be an existing absolute directory." >&2
    exit 6
  fi
  if [[ "$(readlink -f "${REUSE_CACHE_ROOT}")" == "$(readlink -f "${CACHE_ROOT}")" ]]; then
    echo "[ERROR] Reuse cache root and new cache root must differ." >&2
    exit 6
  fi
  for condition in "${CONDITIONS[@]}"; do
    reuse_complete_cache_dir \
      "${REUSE_CACHE_ROOT}/oea/${condition}" \
      "${CACHE_ROOT}/oea/${condition}"
    reuse_complete_cache_dir \
      "${REUSE_CACHE_ROOT}/vanilla/${condition}" \
      "${CACHE_ROOT}/vanilla/${condition}"
    reuse_complete_cache_dir \
      "${REUSE_CACHE_ROOT}/whisper/${condition}" \
      "${CACHE_ROOT}/whisper/${condition}"
  done
  reuse_complete_cache_dir \
    "${REUSE_CACHE_ROOT}/bge/corpus" \
    "${CACHE_ROOT}/bge/corpus"
  for template in none bge_retrieval; do
    reuse_complete_cache_dir \
      "${REUSE_CACHE_ROOT}/bge/dev_${template}" \
      "${CACHE_ROOT}/bge/dev_${template}"
  done
  ACTUAL_REUSE_COUNT=0
  if [[ -f "${RUN_DIR}/reused_cache_dirs.jsonl" ]]; then
    ACTUAL_REUSE_COUNT="$(wc -l < "${RUN_DIR}/reused_cache_dirs.jsonl")"
    ACTUAL_REUSE_COUNT="${ACTUAL_REUSE_COUNT//[[:space:]]/}"
  fi
  {
    printf 'required_reuse_count=%s\n' "${REQUIRED_REUSE_COUNT}"
    printf 'actual_reuse_count=%s\n' "${ACTUAL_REUSE_COUNT}"
  } | tee "${RUN_DIR}/reuse_summary.txt"
  if [[ "${REQUIRED_REUSE_COUNT}" -gt 0 ]] \
    && [[ "${ACTUAL_REUSE_COUNT}" -ne "${REQUIRED_REUSE_COUNT}" ]]; then
    echo "[ERROR] Required ${REQUIRED_REUSE_COUNT} immutable cache reuses; observed ${ACTUAL_REUSE_COUNT}." >&2
    exit 6
  fi
elif [[ "${REQUIRED_REUSE_COUNT}" -gt 0 ]]; then
  echo "[ERROR] PHASE2_REQUIRED_REUSE_COUNT requires PHASE2_REUSE_CACHE_ROOT." >&2
  exit 6
fi
run_step resolve_oea \
  python scripts/build_official_oea_eval_config.py \
    --protocol-config "${OEA_PROTOCOL}" \
    --model-lock "${OEA_LOCK}" \
    --output "${OEA_RESOLVED}"
run_step resolve_vanilla \
  python scripts/build_vanilla_backbone_eval_config.py \
    --protocol-config "${VANILLA_PROTOCOL}" \
    --model-lock "${VANILLA_LOCK}" \
    --output "${VANILLA_RESOLVED}"

for condition in "${CONDITIONS[@]}"; do
  content="audio_only"
  if [[ "${condition}" == "clean" ]]; then
    content="both"
  fi
  run_cache_step "oea_${condition}" "${CACHE_ROOT}/oea/${condition}" \
    python scripts/generate_asrur_omni_caches.py \
      --mode oea \
      --main-config "${MAIN_CONFIG}" \
      --resolved-model-config "${OEA_RESOLVED}" \
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
  run_cache_step "vanilla_${condition}" "${CACHE_ROOT}/vanilla/${condition}" \
    python scripts/generate_asrur_omni_caches.py \
      --mode original_omni \
      --main-config "${MAIN_CONFIG}" \
      --resolved-model-config "${VANILLA_RESOLVED}" \
      --model-root "${OEA_MODEL_ROOT}" \
      --corpus "${CORPUS}" \
      --audio-manifest "${SQuTR_MANIFEST}" \
      --subset fiqa \
      --conditions "${condition}" \
      --text-batch-size "${TEXT_BATCH_SIZE}" \
      --audio-batch-size "${AUDIO_BATCH_SIZE}" \
      --content "${content}" \
      --output-dir "${CACHE_ROOT}/vanilla/${condition}" \
      "${DRY_ARGUMENTS[@]}"
done

run_cache_step bge_corpus "${CACHE_ROOT}/bge/corpus" \
  python scripts/generate_asrur_frozen_caches.py bge \
    --config "${MAIN_CONFIG}" \
    --output-dir "${CACHE_ROOT}/bge/corpus" \
    --dataset FiQA \
    --split corpus \
    --device cuda:0 \
    --dtype bfloat16 \
    --input "${CORPUS}" \
    --input-kind corpus \
    --query-template none \
    --batch-size "${BGE_BATCH_SIZE}" \
    "${DRY_ARGUMENTS[@]}"
for template in none bge_retrieval; do
  run_cache_step "bge_dev_${template}" "${CACHE_ROOT}/bge/dev_${template}" \
    python scripts/generate_asrur_frozen_caches.py bge \
      --config "${MAIN_CONFIG}" \
      --output-dir "${CACHE_ROOT}/bge/dev_${template}" \
      --dataset FiQA \
      --split dev \
      --device cuda:0 \
      --dtype bfloat16 \
      --input "${QUERIES}" \
      --input-kind queries \
      --query-qrels "${DEV_QRELS}" \
      --query-template "${template}" \
      --batch-size "${BGE_BATCH_SIZE}" \
      "${DRY_ARGUMENTS[@]}"
done
for condition in "${CONDITIONS[@]}"; do
  WHISPER_RESUME_ARGUMENTS=()
  if [[ -n "${WHISPER_RESUME_SOURCE_ROOT}" ]]; then
    WHISPER_SOURCE_DIR="${WHISPER_RESUME_SOURCE_ROOT}/whisper/${condition}"
    if [[ -d "${WHISPER_SOURCE_DIR}" && ! -f "${WHISPER_SOURCE_DIR}/cache_manifest.json" ]]; then
      WHISPER_RESUME_ARGUMENTS=(
        --resume-shards-from "${WHISPER_SOURCE_DIR}"
        --resume-source-git-commit "${WHISPER_RESUME_SOURCE_GIT_COMMIT}"
      )
    fi
  fi
  run_cache_step "whisper_${condition}" "${CACHE_ROOT}/whisper/${condition}" \
    python scripts/generate_asrur_frozen_caches.py whisper \
      --config "${MAIN_CONFIG}" \
      --output-dir "${CACHE_ROOT}/whisper/${condition}" \
      --dataset SQuTR-FiQA \
      --split "test_${condition}" \
      --device cuda:0 \
      --dtype bfloat16 \
      --input "${SQuTR_MANIFEST}" \
      --subset fiqa \
      --conditions "${condition}" \
      --max-record-attempts "${WHISPER_MAX_RECORD_ATTEMPTS}" \
      "${WHISPER_RESUME_ARGUMENTS[@]}" \
      "${DRY_ARGUMENTS[@]}"
done

if [[ "${MODE}" == "--dry-run" ]]; then
  echo "[INFO] Phase-2 dry-run complete."
  echo "[INFO] Dense ranking and test encodings intentionally wait for formal dev selection."
  echo "[INFO] Run directory: ${RUN_DIR}"
  exit 0
fi

mkdir -p "${CACHE_ROOT}/rankings/bge_dev"
for template in none bge_retrieval; do
  run_step "dense_dev_${template}" \
    python scripts/generate_asrur_frozen_caches.py dense \
      --config "${MAIN_CONFIG}" \
      --query-embeddings "${CACHE_ROOT}/bge/dev_${template}/embeddings.npy" \
      --query-ids "${CACHE_ROOT}/bge/dev_${template}/ids.jsonl" \
      --query-id-field id \
      --document-embeddings "${CACHE_ROOT}/bge/corpus/embeddings.npy" \
      --document-ids "${CACHE_ROOT}/bge/corpus/ids.jsonl" \
      --document-id-field id \
      --output "${CACHE_ROOT}/rankings/bge_dev/${template}.jsonl"
done
run_step select_bge_template \
  python scripts/select_asrur_bge_query_template.py \
    --ranking "none=${CACHE_ROOT}/rankings/bge_dev/none.jsonl" \
    --ranking "bge_retrieval=${CACHE_ROOT}/rankings/bge_dev/bge_retrieval.jsonl" \
    --qrels "${DEV_QRELS}" \
    --output-dir "${CACHE_ROOT}/bge_template_selection"

SELECTED_TEMPLATE="$(
  python - "${CACHE_ROOT}/bge_template_selection/frozen_selection.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = value["selected_query_template"]
if selected not in {"none", "bge_retrieval"}:
    raise ValueError(selected)
print(selected)
PY
)"
echo "SELECTED_BGE_QUERY_TEMPLATE=${SELECTED_TEMPLATE}"

run_step bge_gold_test \
  python scripts/generate_asrur_frozen_caches.py bge \
    --config "${MAIN_CONFIG}" \
    --output-dir "${CACHE_ROOT}/bge/gold_test" \
    --dataset FiQA \
    --split test \
    --device cuda:0 \
    --dtype bfloat16 \
    --input "${QUERIES}" \
    --input-kind queries \
    --query-qrels "${TEST_QRELS}" \
    --query-template "${SELECTED_TEMPLATE}" \
    --batch-size "${BGE_BATCH_SIZE}"
run_step dense_gold_test \
  python scripts/generate_asrur_frozen_caches.py dense \
    --config "${MAIN_CONFIG}" \
    --query-embeddings "${CACHE_ROOT}/bge/gold_test/embeddings.npy" \
    --query-ids "${CACHE_ROOT}/bge/gold_test/ids.jsonl" \
    --query-id-field id \
    --document-embeddings "${CACHE_ROOT}/bge/corpus/embeddings.npy" \
    --document-ids "${CACHE_ROOT}/bge/corpus/ids.jsonl" \
    --document-id-field id \
    --output "${CACHE_ROOT}/rankings/bge_gold_test.jsonl"

mkdir -p \
  "${CACHE_ROOT}/rankings/oea" \
  "${CACHE_ROOT}/rankings/vanilla" \
  "${CACHE_ROOT}/rankings/bge_asr1"
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
  run_step "dense_vanilla_${condition}" \
    python scripts/generate_asrur_frozen_caches.py dense \
      --config "${MAIN_CONFIG}" \
      --query-embeddings "${CACHE_ROOT}/vanilla/${condition}/audio_embeddings.npy" \
      --query-ids "${CACHE_ROOT}/vanilla/${condition}/audio_ids.jsonl" \
      --query-id-field id \
      --document-embeddings "${CACHE_ROOT}/vanilla/clean/document_embeddings.npy" \
      --document-ids "${CACHE_ROOT}/vanilla/clean/document_ids.jsonl" \
      --document-id-field id \
      --output "${CACHE_ROOT}/rankings/vanilla/${condition}.jsonl"
  run_step "bge_asr1_${condition}" \
    python scripts/generate_asrur_frozen_caches.py bge \
      --config "${MAIN_CONFIG}" \
      --output-dir "${CACHE_ROOT}/bge/asr1_${condition}" \
      --dataset SQuTR-FiQA \
      --split "test_${condition}" \
      --device cuda:0 \
      --dtype bfloat16 \
      --input "${CACHE_ROOT}/whisper/${condition}/nbest.jsonl" \
      --input-kind nbest1 \
      --query-template "${SELECTED_TEMPLATE}" \
      --batch-size "${BGE_BATCH_SIZE}"
  run_step "dense_asr1_${condition}" \
    python scripts/generate_asrur_frozen_caches.py dense \
      --config "${MAIN_CONFIG}" \
      --query-embeddings "${CACHE_ROOT}/bge/asr1_${condition}/embeddings.npy" \
      --query-ids "${CACHE_ROOT}/bge/asr1_${condition}/ids.jsonl" \
      --query-id-field id \
      --document-embeddings "${CACHE_ROOT}/bge/corpus/embeddings.npy" \
      --document-ids "${CACHE_ROOT}/bge/corpus/ids.jsonl" \
      --document-id-field id \
      --output "${CACHE_ROOT}/rankings/bge_asr1/${condition}.jsonl"

  METRICS_DIR="${RESULT_ROOT}/${condition}"
  if [[ -f "${METRICS_DIR}/metrics.json" ]]; then
    echo "[INFO] Reusing existing dense metrics: ${METRICS_DIR}/metrics.json"
  else
    run_step "evaluate_${condition}" \
      python scripts/evaluate_asrur_dense_rankings.py \
        --ranking "B1_whisper_1best_bge_dense=${CACHE_ROOT}/rankings/bge_asr1/${condition}.jsonl" \
        --ranking "B2_original_omni=${CACHE_ROOT}/rankings/vanilla/${condition}.jsonl" \
        --ranking "B3_oea=${CACHE_ROOT}/rankings/oea/${condition}.jsonl" \
        --ranking "U1_gold_bge_dense=${CACHE_ROOT}/rankings/bge_gold_test.jsonl" \
        --qrels "${TEST_QRELS}" \
        --output-dir "${METRICS_DIR}"
  fi
done

python - "${RUN_DIR}" "${CACHE_ROOT}" "${RESULT_ROOT}" "${SELECTED_TEMPLATE}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir, cache_root, result_root = map(Path, sys.argv[1:4])
conditions = ("clean", "snr_20", "snr_10", "snr_0")
metrics = {
    condition: str((result_root / condition / "metrics.json").resolve())
    for condition in conditions
}
payload = {
    "schema_version": 1,
    "status": "complete",
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "cache_root": str(cache_root.resolve()),
    "result_root": str(result_root.resolve()),
    "selected_bge_query_template": sys.argv[4],
    "condition_metrics": metrics,
}
(run_dir / "completion_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

df -hT "$(dirname "${CACHE_ROOT}")" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
nvidia-smi > "${RUN_DIR}/gpu_final.txt"
echo "[INFO] Phase-2 frozen retrieval completed."
echo "[INFO] Cache root: ${CACHE_ROOT}"
echo "[INFO] Result root: ${RESULT_ROOT}"
echo "[INFO] Run directory: ${RUN_DIR}"
