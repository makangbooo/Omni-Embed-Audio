#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <pairing-audit-root> [variant ...]" >&2
  exit 2
fi

PAIRING_ROOT="$(realpath "$1")"
shift
if [[ "$#" -eq 0 ]]; then
  VARIANTS=(oea_nemo3b oea_nemo3b_cl oea_qwen3b oea_qwen3b_cl oea_qwen7b oea_qwen7b_cl)
else
  VARIANTS=("$@")
fi

echo "EXPERIMENT_NAME=Six OEA models negative UIQ Tables 4 and 17"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "MODEL_COUNT=${#VARIANTS[@]}"
echo "TOTAL_WORKLOAD=$((${#VARIANTS[@]} * 1581)) negative text embeddings; three datasets per model"
echo "ESTIMATED_TOTAL_TIME=20-60 minutes on RTX 4090"
echo "DOWNLOADS_REQUIRED=no"
echo "OVERWRITE_DELETE_RISK=none; each model uses a timestamped output"

for variant in "${VARIANTS[@]}"; do
  echo "MATRIX_STAGE_START=${variant}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    bash scripts/run_oea_negative_uiq.sh "${variant}" "${PAIRING_ROOT}"
  rc=$?
  echo "MATRIX_STAGE_END=${variant} RC=${rc}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "MATRIX_STATUS=failed"
    echo "FAILED_VARIANT=${variant}"
    exit "${rc}"
  fi
done

COLLECT_ARGS=()
for variant in "${VARIANTS[@]}"; do
  COLLECT_ARGS+=(--variant "${variant}")
done
python scripts/collect_oea_negative_uiq_results.py \
  --expected-git-commit "$(git rev-parse HEAD)" "${COLLECT_ARGS[@]}" || exit $?
echo "MATRIX_STATUS=complete"
echo "COMPLETED_VARIANTS=${VARIANTS[*]}"
