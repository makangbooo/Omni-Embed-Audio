#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="negative_uiq_exact_pairing_audit_${STAMP}"
RESULT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_TIME="$(date -Is)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --short)"

echo "EXPERIMENT_NAME=Negative UIQ deterministic-caption pairing audit"
echo "PAPER_EXPERIMENTS=EXP-16/17 Tables 4 and 17"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GPU_USED=no"
echo "OEA_OFFICIAL_SOURCE_USED=no; OEA official released UIQ JSONL data used"
echo "TOTAL_WORKLOAD=1,581 negative UIQ rows across Clotho, AudioCaps, and MECAT"
echo "ESTIMATED_TOTAL_TIME=under 1 minute"
echo "DOWNLOADS_REQUIRED=no"
echo "OVERWRITE_DELETE_RISK=none; timestamped output"
echo "RESULT_DIRECTORY=${RESULT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

if [[ -n "${GIT_STATUS}" ]]; then
  echo "[ERROR] Formal pairing audit requires a clean Git worktree: ${GIT_STATUS}" >&2
  exit 2
fi

declare -a DATASETS=(clotho audiocaps mecat)
declare -A NEGATIVE_PATHS=(
  [clotho]="data/UIQ/clotho/clotho_evaluation_negative_queries.jsonl"
  [audiocaps]="data/UIQ/audiocaps/audiocaps_test_negative_queries.jsonl"
  [mecat]="data/UIQ/mecat/mecat_negative_queries.jsonl"
)
declare -A POSITIVE_PATHS=(
  [clotho]="data/UIQ/clotho/clotho_evaluation_question_queries.jsonl"
  [audiocaps]="data/UIQ/audiocaps/audiocaps_test_question_queries.jsonl"
  [mecat]="data/UIQ/mecat/mecat_question_queries.jsonl"
)

FINAL_RC=0
for dataset in "${DATASETS[@]}"; do
  echo "STAGE_START=pairing_${dataset}"
  python scripts/reconstruct_negative_uiq_pairings.py \
    --dataset "${dataset}" \
    --negative-jsonl "${NEGATIVE_PATHS[${dataset}]}" \
    --reference-positive-jsonl "${POSITIVE_PATHS[${dataset}]}" \
    --output-dir "${RESULT_DIR}/${dataset}"
  RC=$?
  echo "STAGE_END=pairing_${dataset} RC=${RC}"
  if [[ "${RC}" -ne 0 ]]; then
    FINAL_RC="${RC}"
  fi
done

export RESULT_DIR GIT_COMMIT START_TIME FINAL_RC
python - <<'PY'
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

root = Path(os.environ["RESULT_DIR"])
datasets = {}
for dataset in ("clotho", "audiocaps", "mecat"):
    path = root / dataset / "reconstruction_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    datasets[dataset] = {
        "status": report["status"],
        "negative_row_count": report["negative_row_count"],
        "matched_pairing_count": report["matched_pairing_count"],
        "failed_pairing_count": report["failed_pairing_count"],
        "full_coverage": report["full_coverage"],
        "target_match_methods": report["target_match_methods"],
        "hard_negative_match_methods": report["hard_negative_match_methods"],
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
summary = {
    "schema_version": 1,
    "status": "complete" if all(v["full_coverage"] for v in datasets.values()) else "incomplete",
    "git_commit": os.environ["GIT_COMMIT"],
    "started_at": os.environ["START_TIME"],
    "finished_at": datetime.now().astimezone().isoformat(),
    "strict_paper_pairing_reproduction": False,
    "pairing_source": "INFERRED_DETERMINISTIC_RELEASED_CAPTION_IDENTITY",
    "datasets": datasets,
}
path = root / "summary.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("PAIRING_AUDIT_SUMMARY=" + json.dumps(summary, sort_keys=True))
print("SUMMARY_PATH=" + str(path))
print("SUMMARY_SHA256=" + hashlib.sha256(path.read_bytes()).hexdigest())
PY

echo "FINAL_RUN_RC=${FINAL_RC}"
echo "END_TIME=$(date -Is)"
exit "${FINAL_RC}"
