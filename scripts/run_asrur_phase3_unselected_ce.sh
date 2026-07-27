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
MAIN_CONFIG="${ROOT_DIR}/configs/asr_uncertainty_reranking/main_experiment.json"
MODEL_RESOURCE_MANIFEST="${ROOT_DIR}/configs/asr_uncertainty_reranking/resources/models.json"
CORPUS="${DATA_ROOT}/fiqa_mteb/corpus.jsonl"
QUERIES="${DATA_ROOT}/fiqa_mteb/queries.jsonl"
TEST_QRELS="${DATA_ROOT}/fiqa_mteb/qrels/test.jsonl"
PHASE2_CACHE_ROOT="${PHASE2_CACHE_ROOT:?Set PHASE2_CACHE_ROOT to the completed immutable Phase-2 cache}"
PHASE2_AUDIT="${PHASE2_AUDIT:?Set PHASE2_AUDIT to the completed Phase-2 Go/No-Go JSON}"
GIT_COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="${GIT_COMMIT:0:12}"
PHASE3_CACHE_ROOT="${PHASE3_CACHE_ROOT:-/home/jg525/experiment_cache/asr_uncertainty/fiqa_phase3_unselected_${COMMIT_SHORT}}"
PHASE3_RESULT_ROOT="${PHASE3_RESULT_ROOT:-${ROOT_DIR}/results/raw/asrur_phase3_unselected_${COMMIT_SHORT}}"
RUN_ID="asrur_phase3_unselected_ce_${MODE#--}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
CONDITIONS=(clean snr_20 snr_10 snr_0)
CE_BATCH_SIZE="${ASRUR_CE_BATCH_SIZE:-16}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[ERROR] Formal Phase-3 runner requires a clean Git worktree." >&2
  git status --short --untracked-files=all >&2
  exit 3
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [[ ! -f "${PHASE2_AUDIT}" ]]; then
  echo "[ERROR] Missing Phase-2 audit: ${PHASE2_AUDIT}" >&2
  exit 4
fi
python - "${PHASE2_AUDIT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("status") != "complete":
    raise SystemExit("Phase-2 audit is not complete")
if value.get("overall_decision") != "GO":
    raise SystemExit(
        "Phase-2 decision is not GO; candidate generation may not be changed "
        "and Phase 3 requires an explicit user decision"
    )
if value.get("candidate_generation_change_authorized") is not False:
    raise SystemExit("Phase-2 audit candidate-generation policy is invalid")
PY

for input in \
  "${MAIN_CONFIG}" \
  "${MODEL_RESOURCE_MANIFEST}" \
  "${CORPUS}" \
  "${QUERIES}" \
  "${TEST_QRELS}"
do
  if [[ ! -f "${input}" ]]; then
    echo "[ERROR] Missing fixed input: ${input}" >&2
    exit 4
  fi
done
for condition in "${CONDITIONS[@]}"; do
  for input in \
    "${PHASE2_CACHE_ROOT}/rankings/oea/${condition}.jsonl" \
    "${PHASE2_CACHE_ROOT}/whisper/${condition}/nbest.jsonl"
  do
    if [[ ! -f "${input}" ]]; then
      echo "[ERROR] Missing completed Phase-2 artifact: ${input}" >&2
      exit 4
    fi
  done
done
if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse run directory: ${RUN_DIR}" >&2
  exit 4
fi
mkdir -p "${RUN_DIR}/steps" "${PHASE3_CACHE_ROOT}" "${PHASE3_RESULT_ROOT}"

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

{
  printf 'git_commit=%s\n' "${GIT_COMMIT}"
  printf 'mode=%s\n' "${MODE}"
  printf 'phase2_cache_root=%s\n' "${PHASE2_CACHE_ROOT}"
  printf 'phase2_audit=%s\n' "${PHASE2_AUDIT}"
  printf 'phase3_cache_root=%s\n' "${PHASE3_CACHE_ROOT}"
  printf 'phase3_result_root=%s\n' "${PHASE3_RESULT_ROOT}"
  printf 'ce_batch_size=%s\n' "${CE_BATCH_SIZE}"
  printf 'asr_pair_count=%s\n' "$((4 * 648 * 100 * 4))"
  printf 'gold_pair_count=%s\n' "$((4 * 648 * 100))"
  printf 'total_pair_count=%s\n' "$((4 * 648 * 100 * 5))"
} > "${RUN_DIR}/run_identity.txt"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short --untracked-files=all > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"

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
    echo "[ERROR] Approved Phase-3 hardware is one RTX 4090; observed: ${GPU_NAME}" >&2
    exit 5
  fi
  run_step d2_d4_strict_audit \
    python scripts/verify_asrur_model_assets.py \
      --manifest "${MODEL_RESOURCE_MANIFEST}" \
      --model-root "${MODELS_ROOT}" \
      --output "${RUN_DIR}/d2_d4_model_audit.json"
fi

GOLD_NBEST_ROOT="${PHASE3_CACHE_ROOT}/gold_nbest"
run_step build_gold_nbest \
  python scripts/build_asrur_gold_nbest.py \
    --queries "${QUERIES}" \
    --qrels "${TEST_QRELS}" \
    --output-dir "${GOLD_NBEST_ROOT}" \
    --dataset FiQA \
    --split test \
    --expected-query-count 648

for condition in "${CONDITIONS[@]}"; do
  CE_ROOT="${PHASE3_CACHE_ROOT}/ce/${condition}"
  GOLD_CE_ROOT="${PHASE3_CACHE_ROOT}/gold_ce/${condition}"
  run_step "ce_${condition}" \
    python scripts/generate_asrur_frozen_caches.py ce \
      --config "${MAIN_CONFIG}" \
      --output-dir "${CE_ROOT}" \
      --dataset SQuTR-FiQA \
      --split "test_${condition}" \
      --device cuda:0 \
      --dtype bfloat16 \
      --candidates "${PHASE2_CACHE_ROOT}/rankings/oea/${condition}.jsonl" \
      --nbest "${PHASE2_CACHE_ROOT}/whisper/${condition}/nbest.jsonl" \
      --corpus "${CORPUS}" \
      --batch-size "${CE_BATCH_SIZE}" \
      --expected-hypotheses 4 \
      "${DRY_ARGUMENTS[@]}"

  run_step "gold_ce_${condition}" \
    python scripts/generate_asrur_frozen_caches.py ce \
      --config "${MAIN_CONFIG}" \
      --output-dir "${GOLD_CE_ROOT}" \
      --dataset SQuTR-FiQA \
      --split "test_${condition}_gold_upper_bound" \
      --device cuda:0 \
      --dtype bfloat16 \
      --candidates "${PHASE2_CACHE_ROOT}/rankings/oea/${condition}.jsonl" \
      --nbest "${GOLD_NBEST_ROOT}/nbest.jsonl" \
      --corpus "${CORPUS}" \
      --batch-size "${CE_BATCH_SIZE}" \
      --expected-hypotheses 1 \
      "${DRY_ARGUMENTS[@]}"

  if [[ "${MODE}" == "--execute" ]]; then
    METRICS_ROOT="${PHASE3_RESULT_ROOT}/${condition}"
    if [[ -f "${METRICS_ROOT}/metrics.json" ]]; then
      echo "[INFO] Reusing existing unselected metrics: ${METRICS_ROOT}/metrics.json"
    else
      run_step "evaluate_${condition}" \
        python scripts/evaluate_asrur_unselected_ce_baselines.py \
          --config "${MAIN_CONFIG}" \
          --top100 "${PHASE2_CACHE_ROOT}/rankings/oea/${condition}.jsonl" \
          --nbest "${PHASE2_CACHE_ROOT}/whisper/${condition}/nbest.jsonl" \
          --cross-encoder "${CE_ROOT}/cross_encoder_scores.jsonl" \
          --gold-cross-encoder "${GOLD_CE_ROOT}/cross_encoder_scores.jsonl" \
          --qrels "${TEST_QRELS}" \
          --output-dir "${METRICS_ROOT}"
    fi
  fi
done

if [[ "${MODE}" == "--dry-run" ]]; then
  echo "[INFO] Phase-3 unselected CE dry-run complete."
else
  python - "${RUN_DIR}" "${PHASE3_CACHE_ROOT}" "${PHASE3_RESULT_ROOT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir, cache_root, result_root = map(Path, sys.argv[1:4])
conditions = ("clean", "snr_20", "snr_10", "snr_0")
(run_dir / "completion_manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "status": "complete",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "cache_root": str(cache_root.resolve()),
            "result_root": str(result_root.resolve()),
            "methods": [
                "B4_1best_ce",
                "B7a_4best_equal",
                "B7b_4best_max",
                "U2_gold_ce",
            ],
            "condition_metrics": {
                condition: str(
                    (result_root / condition / "metrics.json").resolve()
                )
                for condition in conditions
            },
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
  nvidia-smi > "${RUN_DIR}/gpu_final.txt"
  echo "[INFO] Phase-3 unselected CE execution complete."
fi
echo "[INFO] Cache root: ${PHASE3_CACHE_ROOT}"
echo "[INFO] Result root: ${PHASE3_RESULT_ROOT}"
echo "[INFO] Run directory: ${RUN_DIR}"
