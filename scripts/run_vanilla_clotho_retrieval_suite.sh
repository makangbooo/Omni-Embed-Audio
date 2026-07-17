#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <vanilla_backbone_id> <completed-embedding-directory>" >&2
  exit 2
fi

BACKBONE_ID=$1
case "${BACKBONE_ID}" in
  vanilla_nemotron_3b)
    CONFIG_STEM="vanilla_nemotron_3b"
    ;;
  vanilla_qwen2_5_omni_3b)
    CONFIG_STEM="vanilla_qwen2_5_omni_3b"
    ;;
  vanilla_qwen2_5_omni_7b)
    CONFIG_STEM="vanilla_qwen2_5_omni_7b"
    ;;
  *)
    echo "[ERROR] Unknown vanilla backbone: ${BACKBONE_ID}" >&2
    exit 2
    ;;
esac

EMBEDDING_DIR="$(cd "$2" && pwd)"
CONFIG="${ROOT_DIR}/configs/eval/${CONFIG_STEM}_clotho_retrieval_suite.json"
SUITE_ID="${SUITE_ID:-${CONFIG_STEM}_clotho_retrieval_suite_seed42_$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${RESULT_ROOT:-${ROOT_DIR}/results/raw}/${SUITE_ID}"
ATTEMPT_ID="attempt_$(date +%Y%m%d_%H%M%S)"
ATTEMPT_DIR="${SUITE_DIR}/attempts/${ATTEMPT_ID}"

if [[ -e "${ATTEMPT_DIR}" ]]; then
  echo "[ERROR] Attempt directory exists; refusing to overwrite: ${ATTEMPT_DIR}" >&2
  exit 3
fi
mkdir -p "${ATTEMPT_DIR}"
record_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${ATTEMPT_DIR}/exit_code.txt"
}
trap record_exit EXIT

{
  printf 'SUITE_ID=%q RESULT_ROOT=%q ' \
    "${SUITE_ID}" "${RESULT_ROOT:-${ROOT_DIR}/results/raw}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${ATTEMPT_DIR}/command.sh"
git rev-parse HEAD > "${ATTEMPT_DIR}/git_commit.txt"
git status --short > "${ATTEMPT_DIR}/git_status.txt"
if [[ -s "${ATTEMPT_DIR}/git_status.txt" ]]; then
  echo "[ERROR] Formal vanilla retrieval requires a clean Git worktree." >&2
  exit 4
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES=""

{
  printf 'timestamp=%s\n' "$(date -Is)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'conda_prefix=%s\n' "${CONDA_PREFIX:-}"
  printf 'CUDA_VISIBLE_DEVICES=disabled\n'
  python --version
  python -c 'import numpy; print(f"numpy={numpy.__version__}")'
} > "${ATTEMPT_DIR}/environment.txt" 2>&1

exec > >(tee "${ATTEMPT_DIR}/stdout.log") \
  2> >(tee "${ATTEMPT_DIR}/stderr.log" >&2)

echo "[INFO] Backbone: ${BACKBONE_ID}"
echo "[INFO] Suite ID: ${SUITE_ID}"
echo "[INFO] Suite directory: ${SUITE_DIR}"
echo "[INFO] Attempt directory: ${ATTEMPT_DIR}"
echo "[INFO] Embedding directory: ${EMBEDDING_DIR}"
echo "[INFO] Base-only evidence path: no OEA checkpoint, LoRA, or projection head"
echo "[INFO] GPU disabled"

python scripts/prepare_embedding_evaluation_suite.py prepare \
  --config "${CONFIG}" \
  --embedding-dir "${EMBEDDING_DIR}" \
  --suite-dir "${SUITE_DIR}"

while IFS=$'\t' read -r protocol_id protocol_config output_dir; do
  if [[ -f "${output_dir}/metrics.json" ]]; then
    if python - "${output_dir}/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if report.get("status") == "complete" else 1)
PY
    then
      echo "[INFO] Reusing complete protocol: ${protocol_id}"
      continue
    fi
    echo "[ERROR] Protocol has non-complete prior artifacts: ${protocol_id}" >&2
    echo "[ERROR] Use a new SUITE_ID; no artifacts were overwritten." >&2
    exit 5
  fi
  if [[ -d "${output_dir}" ]] && \
     [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[ERROR] Protocol output directory is non-empty: ${output_dir}" >&2
    exit 6
  fi
  echo "[INFO] Evaluating protocol: ${protocol_id}"
  bash scripts/run_embedding_evaluation.sh \
    "${protocol_config}" \
    "${output_dir}"
done < <(
  python - "${SUITE_DIR}/suite_plan.json" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for protocol in plan["protocols"]:
    fields = (
        protocol["protocol_id"],
        protocol["config"],
        protocol["output_dir"],
    )
    if any("\t" in value or "\n" in value for value in fields):
        raise ValueError("suite plan contains an unsafe shell field")
    print("\t".join(fields))
PY
)

python scripts/prepare_embedding_evaluation_suite.py finalize \
  --config "${CONFIG}" \
  --embedding-dir "${EMBEDDING_DIR}" \
  --suite-dir "${SUITE_DIR}"

echo "[INFO] Vanilla retrieval suite completed"
echo "[INFO] Summary: ${SUITE_DIR}/retrieval_summary.csv"
echo "[INFO] Metrics: ${SUITE_DIR}/suite_metrics.json"
