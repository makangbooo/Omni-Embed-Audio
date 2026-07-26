#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 <fiqa|models> [--dry-run]" >&2
  exit 2
fi

BUNDLE=$1
MODE=${2:-}
if [[ -n "${MODE}" && "${MODE}" != "--dry-run" ]]; then
  echo "[ERROR] Optional mode must be exactly --dry-run: ${MODE}" >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="${ASRUR_ENV_NAME:-oea-repro}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/home/jg525/model_cache}"
HF_HOME="${HF_HOME:-${MODEL_CACHE_ROOT}/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
DOWNLOAD_MAX_ATTEMPTS="${ASRUR_DOWNLOAD_MAX_ATTEMPTS:-20}"
DOWNLOAD_RETRY_BACKOFF_SECONDS="${ASRUR_DOWNLOAD_RETRY_BACKOFF_SECONDS:-30}"
DOWNLOAD_MAX_RETRY_DELAY_SECONDS="${ASRUR_DOWNLOAD_MAX_RETRY_DELAY_SECONDS:-600}"
DOWNLOAD_MAX_WORKERS="${ASRUR_DOWNLOAD_MAX_WORKERS:-1}"

case "${BUNDLE}" in
  fiqa)
    RUN_PREFIX="asrur_d1_fiqa"
    MANIFEST="${ROOT_DIR}/configs/asr_uncertainty_reranking/resources/fiqa.json"
    DESTINATION_ROOT="${DATA_ROOT}"
    ;;
  models)
    RUN_PREFIX="asrur_d2_d4_models"
    MANIFEST="${ROOT_DIR}/configs/asr_uncertainty_reranking/resources/models.json"
    DESTINATION_ROOT="${MODEL_CACHE_ROOT}"
    ;;
  *)
    echo "[ERROR] Bundle must be exactly 'fiqa' or 'models': ${BUNDLE}" >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_PREFIX}_download_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
if [[ "${MODE}" != "--dry-run" ]]; then
  mkdir -p "${DESTINATION_ROOT}" "${HF_HUB_CACHE}"
fi

record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
}
record_signal() {
  local signal_name=$1
  local exit_code=$2
  printf '%s\n' "${signal_name}" > "${RUN_DIR}/termination_signal.txt"
  exit "${exit_code}"
}
trap record_wrapper_exit EXIT
trap 'record_signal SIGHUP 129' HUP
trap 'record_signal SIGINT 130' INT
trap 'record_signal SIGTERM 143' TERM

{
  printf 'DATA_ROOT=%q ' "${DATA_ROOT}"
  printf 'MODEL_CACHE_ROOT=%q ' "${MODEL_CACHE_ROOT}"
  printf 'HF_HOME=%q ' "${HF_HOME}"
  printf 'HF_HUB_CACHE=%q ' "${HF_HUB_CACHE}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short --untracked-files=all > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
df -hT "$(dirname "${DESTINATION_ROOT}")" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

if [[ -s "${RUN_DIR}/git_status.txt" ]]; then
  echo "[ERROR] Resource download requires a clean Git worktree." >&2
  cat "${RUN_DIR}/git_status.txt" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] Missing resource manifest: ${MANIFEST}" >&2
  exit 3
fi
if [[ "${MODE}" != "--dry-run" ]]; then
  if ! command -v flock >/dev/null 2>&1; then
    echo "[ERROR] flock is required for the shared-cache concurrency guard." >&2
    exit 3
  fi
  LOCK_FILE="${DESTINATION_ROOT}/.asrur_${BUNDLE}_download.lock"
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "[ERROR] Another ${BUNDLE} download holds ${LOCK_FILE}." >&2
    exit 4
  fi
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES=""
export HF_HOME
export HF_HUB_CACHE
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Bundle: ${BUNDLE}"
echo "[INFO] Manifest: ${MANIFEST}"
echo "[INFO] Destination root: ${DESTINATION_ROOT}"
echo "[INFO] HF endpoint: ${HF_ENDPOINT:-https://huggingface.co}"
echo "[INFO] HF Xet disabled: ${HF_HUB_DISABLE_XET}"
echo "[INFO] Download attempts: ${DOWNLOAD_MAX_ATTEMPTS}"
echo "[INFO] Download workers: ${DOWNLOAD_MAX_WORKERS}"
echo "[INFO] GPU disabled"
echo "[INFO] User approval recorded in the immutable manifest"

if [[ "${MODE}" == "--dry-run" ]]; then
  python - "${MANIFEST}" "${DESTINATION_ROOT}" "${RUN_DIR}/dry_run_plan.json" <<'PY'
import json
import shutil
import sys
from pathlib import Path, PurePosixPath

manifest_path = Path(sys.argv[1]).resolve()
destination_root = Path(sys.argv[2]).expanduser().resolve()
output = Path(sys.argv[3]).resolve()
specification = json.loads(manifest_path.read_text(encoding="utf-8"))
assets = []
total_bytes = 0
for asset in specification["assets"]:
    patterns = asset["allow_patterns"]
    required = asset["required_files"]
    expected = asset["expected_files"]
    paths = [entry["path"] for entry in expected]
    if patterns != required or patterns != paths:
        raise RuntimeError(f"inventory mismatch for {asset['name']}")
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"duplicate paths for {asset['name']}")
    for value in paths:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe path for {asset['name']}: {value}")
    selected_bytes = sum(int(entry["size_bytes"]) for entry in expected)
    if selected_bytes != int(asset["expected_selected_bytes"]):
        raise RuntimeError(f"selected byte mismatch for {asset['name']}")
    total_bytes += selected_bytes
    assets.append(
        {
            "name": asset["name"],
            "repo_id": asset["repo_id"],
            "repo_type": asset["repo_type"],
            "revision": asset["revision"],
            "destination": str(destination_root / asset["local_subdir"]),
            "selected_file_count": len(paths),
            "selected_bytes": selected_bytes,
        }
    )
declared_total = specification.get("expected_selected_bytes")
if declared_total is not None and int(declared_total) != total_bytes:
    raise RuntimeError("manifest total byte mismatch")
probe = destination_root if destination_root.exists() else destination_root.parent
free_bytes = shutil.disk_usage(probe).free if probe.exists() else None
plan = {
    "status": "dry_run_complete",
    "manifest": str(manifest_path),
    "destination_root": str(destination_root),
    "network_access_performed": False,
    "download_performed": False,
    "assets": assets,
    "selected_bytes": total_bytes,
    "free_bytes": free_bytes,
}
output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(plan, indent=2, sort_keys=True))
PY
  printf '0\n' > "${RUN_DIR}/download_exit_code.txt"
  echo "[INFO] Dry run complete; no network or download was performed"
  echo "[INFO] Audit artifacts: ${RUN_DIR}"
  exit 0
fi

set +e
python scripts/download_model_assets.py \
  --manifest "${MANIFEST}" \
  --model-root "${DESTINATION_ROOT}" \
  --output "${RUN_DIR}/download_manifest.json" \
  --max-download-attempts "${DOWNLOAD_MAX_ATTEMPTS}" \
  --retry-backoff-seconds "${DOWNLOAD_RETRY_BACKOFF_SECONDS}" \
  --max-retry-delay-seconds "${DOWNLOAD_MAX_RETRY_DELAY_SECONDS}" \
  --max-workers "${DOWNLOAD_MAX_WORKERS}"
DOWNLOAD_EXIT_CODE=$?
set -e
printf '%s\n' "${DOWNLOAD_EXIT_CODE}" > "${RUN_DIR}/download_exit_code.txt"

df -hT "${DESTINATION_ROOT}" > "${RUN_DIR}/disk_after.txt" 2>&1 || true
du -sh "${DESTINATION_ROOT}" > "${RUN_DIR}/destination_root_size.txt" 2>&1 || true

if [[ "${DOWNLOAD_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] ${BUNDLE} download failed; partial files were preserved for resume." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${DOWNLOAD_EXIT_CODE}"
fi

echo "[INFO] ${BUNDLE} download and selected-file SHA256 verification completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
