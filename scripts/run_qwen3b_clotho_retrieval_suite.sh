#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <completed-embedding-directory>" >&2
  exit 2
fi

EMBEDDING_DIR="$(cd "$1" && pwd)"
CONFIG="${RETRIEVAL_SUITE_CONFIG:-${ROOT_DIR}/configs/eval/qwen3b_cl_clotho_retrieval_suite.json}"
SUITE_PREFIX="${RETRIEVAL_SUITE_PREFIX:-oea_qwen3b_clotho_retrieval_suite_seed42}"
SUITE_ID="${SUITE_ID:-${SUITE_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${RESULT_ROOT:-${ROOT_DIR}/results/raw}/${SUITE_ID}"
ATTEMPT_ID="attempt_$(date +%Y%m%d_%H%M%S)"
ATTEMPT_DIR="${SUITE_DIR}/attempts/${ATTEMPT_ID}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Retrieval suite config is missing: ${CONFIG}" >&2
  exit 3
fi
CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
case "${CONFIG}" in
  "${ROOT_DIR}"/*) ;;
  *)
    echo "[ERROR] Retrieval suite config must be inside the repository: ${CONFIG}" >&2
    exit 3
    ;;
esac
if ! git ls-files --error-unmatch -- "${CONFIG#${ROOT_DIR}/}" >/dev/null 2>&1; then
  echo "[ERROR] Retrieval suite config must be tracked by Git: ${CONFIG}" >&2
  exit 3
fi

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
  printf 'SUITE_ID=%q RESULT_ROOT=%q RETRIEVAL_SUITE_CONFIG=%q ' \
    "${SUITE_ID}" "${RESULT_ROOT:-${ROOT_DIR}/results/raw}" \
    "${CONFIG}"
  printf 'RETRIEVAL_SUITE_PREFIX=%q ' "${SUITE_PREFIX}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${ATTEMPT_DIR}/command.sh"
git rev-parse HEAD > "${ATTEMPT_DIR}/git_commit.txt"
git status --short > "${ATTEMPT_DIR}/git_status.txt"
if [[ -s "${ATTEMPT_DIR}/git_status.txt" ]]; then
  echo "[ERROR] Formal retrieval suite requires a clean Git worktree." >&2
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

SUITE_START_EPOCH="$(date +%s)"

format_duration() {
  local total="$1"
  printf '%02d:%02d:%02d' \
    "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

print_retrieval_progress() {
  local stage="$1"
  local completed="$2"
  local total="$3"
  local stage_started="$4"
  local now overall_elapsed stage_elapsed percent_whole percent_fraction
  local throughput remaining expected_completion
  now="$(date +%s)"
  overall_elapsed="$((now - SUITE_START_EPOCH))"
  stage_elapsed="$((now - stage_started))"
  percent_whole="$((completed * 100 / total))"
  percent_fraction="$(((completed * 10000 / total) % 100))"
  if [[ "${completed}" -gt 0 && "${overall_elapsed}" -gt 0 ]]; then
    throughput="$(awk -v completed="${completed}" -v elapsed="${overall_elapsed}" \
      'BEGIN {printf "%.4f", completed / elapsed}')"
    remaining="$((overall_elapsed * (total - completed) / completed))"
    expected_completion="$(date -Is --date="@$((now + remaining))")"
    remaining="$(format_duration "${remaining}")"
  else
    throughput="0.0000"
    remaining="unknown"
    expected_completion="unknown"
  fi
  printf '[PROGRESS] stage=%s current=%d total=%d percent=%d.%02d%% ' \
    "${stage}" "${completed}" "${total}" \
    "${percent_whole}" "${percent_fraction}"
  printf 'stage_elapsed=%s overall_elapsed=%s throughput=%s_protocols_per_second ' \
    "$(format_duration "${stage_elapsed}")" \
    "$(format_duration "${overall_elapsed}")" "${throughput}"
  printf 'stage_remaining=%s overall_remaining=%s expected_completion=%s\n' \
    "${remaining}" "${remaining}" "${expected_completion}"
}

echo "[INFO] Suite ID: ${SUITE_ID}"
echo "[INFO] Suite directory: ${SUITE_DIR}"
echo "[INFO] Attempt directory: ${ATTEMPT_DIR}"
echo "[INFO] Embedding directory: ${EMBEDDING_DIR}"
echo "[INFO] GPU disabled"

python scripts/prepare_embedding_evaluation_suite.py prepare \
  --config "${CONFIG}" \
  --embedding-dir "${EMBEDDING_DIR}" \
  --suite-dir "${SUITE_DIR}"

TOTAL_PROTOCOLS="$(python - "${SUITE_DIR}/suite_plan.json" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(len(plan["protocols"]))
PY
)"
COMPLETED_PROTOCOLS=0
print_retrieval_progress \
  "suite_prepared" "${COMPLETED_PROTOCOLS}" "${TOTAL_PROTOCOLS}" \
  "${SUITE_START_EPOCH}"

while IFS=$'\t' read -r protocol_id protocol_config output_dir; do
  PROTOCOL_START_EPOCH="$(date +%s)"
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
      COMPLETED_PROTOCOLS="$((COMPLETED_PROTOCOLS + 1))"
      print_retrieval_progress \
        "reused:${protocol_id}" "${COMPLETED_PROTOCOLS}" \
        "${TOTAL_PROTOCOLS}" "${PROTOCOL_START_EPOCH}"
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
  print_retrieval_progress \
    "evaluating:${protocol_id}" "${COMPLETED_PROTOCOLS}" \
    "${TOTAL_PROTOCOLS}" "${PROTOCOL_START_EPOCH}"
  bash scripts/run_embedding_evaluation.sh \
    "${protocol_config}" \
    "${output_dir}"
  COMPLETED_PROTOCOLS="$((COMPLETED_PROTOCOLS + 1))"
  print_retrieval_progress \
    "completed:${protocol_id}" "${COMPLETED_PROTOCOLS}" \
    "${TOTAL_PROTOCOLS}" "${PROTOCOL_START_EPOCH}"
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

print_retrieval_progress \
  "finalized" "${COMPLETED_PROTOCOLS}" "${TOTAL_PROTOCOLS}" \
  "${SUITE_START_EPOCH}"

echo "[INFO] Retrieval suite completed"
echo "[INFO] Summary: ${SUITE_DIR}/retrieval_summary.csv"
echo "[INFO] Metrics: ${SUITE_DIR}/suite_metrics.json"
