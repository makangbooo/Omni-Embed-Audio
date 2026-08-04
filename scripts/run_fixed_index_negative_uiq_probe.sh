#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 || "$2" != "--execute" ]]; then
  echo "Usage: bash scripts/run_fixed_index_negative_uiq_probe.sh <completed-negative-uiq-run-dir> --execute" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] Run this diagnostic inside tmux." >&2
  echo "[INFO] Start one with: tmux new -s fixed_index_probe" >&2
  exit 3
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

SOURCE_RUN="$(realpath "$1")"
for dataset in audiocaps clotho mecat; do
  [[ -f "${SOURCE_RUN}/metrics/${dataset}/metrics.json" ]] || {
    echo "[ERROR] Missing ${dataset} metrics in ${SOURCE_RUN}" >&2
    exit 4
  }
done
[[ -z "$(git status --short --untracked-files=all)" ]] || {
  echo "[ERROR] Formal probe requires a clean Git worktree." >&2
  git status --short --untracked-files=all >&2
  exit 4
}

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="fixed_index_negative_uiq_probe_$(basename "${SOURCE_RUN}")_${STAMP}"
RESULT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/combined.log") 2>&1

echo "EXPERIMENT_NAME=Frozen-index negative UIQ query-adapter probe"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "SOURCE_RUN=${SOURCE_RUN}"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "TRAINABLE_COMPONENT=query-side rank-16 residual adapter only"
echo "AUDIO_EMBEDDINGS=frozen and hash-verified"
echo "AUDIO_INDEX_REBUILT=no"
echo "SPLIT=5-fold connected-component grouping; audio IDs disjoint"
echo "TOTAL_WORKLOAD=5 adapters over 1,581 controlled pairs"
echo "ESTIMATED_TOTAL_TIME=2-8 minutes"
echo "DOWNLOADS_REQUIRED=no"
echo "OEA_OFFICIAL_SOURCE_USED=yes; reuses completed official-source embeddings"
echo "FILE_MODIFICATIONS=new timestamped result and log directories only"
echo "OVERWRITE_DELETE_RISK=none"
echo "RESULT_DIRECTORY=${RESULT_DIR}"
echo "LOG_DIRECTORY=${LOG_DIR}"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv

START_EPOCH="$(date +%s)"
set +e
python scripts/probe_fixed_index_negative_uiq.py \
  --dataset-metrics "${SOURCE_RUN}/metrics/audiocaps/metrics.json" \
  --dataset-metrics "${SOURCE_RUN}/metrics/clotho/metrics.json" \
  --dataset-metrics "${SOURCE_RUN}/metrics/mecat/metrics.json" \
  --output-dir "${RESULT_DIR}" \
  --device cuda
RUN_RC=$?
set -e
echo "FINAL_RUN_RC=${RUN_RC}"
echo "TOTAL_ELAPSED_SECONDS=$(($(date +%s) - START_EPOCH))"
echo "RESULT_DIRECTORY=${RESULT_DIR}"
if [[ "${RUN_RC}" -eq 0 ]]; then
  export RESULT_DIR
  python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RESULT_DIR"]) / "metrics.json"
report = json.loads(path.read_text(encoding="utf-8"))
print("FIXED_INDEX_PROBE_COMPACT=" + json.dumps({
    "status": report["status"],
    "model": report["model"],
    "query_count": report["query_count"],
    "untouched_HNSR": report["overall"]["untouched_oea"]["negative_retrieval"]["HNSR"],
    "adapter_HNSR": report["overall"]["out_of_fold_query_adapter"]["negative_retrieval"]["HNSR"],
    "HNSR_delta": report["overall"]["paired_component_bootstrap_effects"]["HNSR_percentage_points"],
    "margin_delta": report["overall"]["paired_component_bootstrap_effects"]["mean_target_minus_hard_negative_cosine"],
    "pass": report["overall"]["predeclared_pass_rule"],
    "metrics_path": str(path),
}, sort_keys=True))
fi
exit "${RUN_RC}"
