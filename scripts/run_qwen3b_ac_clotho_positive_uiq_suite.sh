#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <completed-caption-embedding-directory> <completed-uiq-embedding-directory>" >&2
  exit 2
fi

CAPTION_EMBEDDING_DIR="$(cd "$1" && pwd)"
UIQ_EMBEDDING_DIR="$(cd "$2" && pwd)"
CONFIG="${ROOT_DIR}/configs/eval/qwen3b_clotho_positive_uiq_suite.json"
SUITE_ID="${SUITE_ID:-oea_qwen3b_ac_clotho_positive_uiq_suite_seed42_$(date +%Y%m%d_%H%M%S)}"
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
  echo "[ERROR] Formal positive UIQ suite requires a clean Git worktree." >&2
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

echo "[INFO] Suite ID: ${SUITE_ID}"
echo "[INFO] Suite directory: ${SUITE_DIR}"
echo "[INFO] Caption embedding directory: ${CAPTION_EMBEDDING_DIR}"
echo "[INFO] UIQ embedding directory: ${UIQ_EMBEDDING_DIR}"
echo "[INFO] GPU disabled"

python scripts/prepare_positive_uiq_evaluation_suite.py prepare \
  --config "${CONFIG}" \
  --caption-embedding-dir "${CAPTION_EMBEDDING_DIR}" \
  --uiq-embedding-dir "${UIQ_EMBEDDING_DIR}" \
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

python scripts/prepare_positive_uiq_evaluation_suite.py finalize \
  --config "${CONFIG}" \
  --caption-embedding-dir "${CAPTION_EMBEDDING_DIR}" \
  --uiq-embedding-dir "${UIQ_EMBEDDING_DIR}" \
  --suite-dir "${SUITE_DIR}"

echo "[INFO] Positive UIQ suite completed"
echo "[INFO] Summary: ${SUITE_DIR}/positive_uiq_summary.csv"
echo "[INFO] Metrics: ${SUITE_DIR}/suite_metrics.json"
