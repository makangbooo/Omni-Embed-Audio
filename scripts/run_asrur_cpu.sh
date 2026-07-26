#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 <validate_config|plan|synthetic_smoke> [output-dir]" >&2
  exit 2
fi

STAGE="$1"
OUTPUT_DIR="${2:-}"
REPOSITORY_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
  pwd
)"
CONFIG="${ASRUR_CONFIG:-${REPOSITORY_ROOT}/configs/asr_uncertainty_reranking/main_experiment.json}"

if [[ "${STAGE}" == "synthetic_smoke" && -z "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] synthetic_smoke requires an output directory" >&2
  exit 3
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] CPU wrapper requires CUDA_VISIBLE_DEVICES to be empty" >&2
  exit 4
fi
export CUDA_VISIBLE_DEVICES=""

PYTHON_BIN="${PYTHON_BIN:-python}"
COMMAND=(
  "${PYTHON_BIN}"
  "${REPOSITORY_ROOT}/scripts/run_asrur_cpu.py"
  --config "${CONFIG}"
  --stage "${STAGE}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  COMMAND+=(--output-dir "${OUTPUT_DIR}")
fi

exec "${COMMAND[@]}"
