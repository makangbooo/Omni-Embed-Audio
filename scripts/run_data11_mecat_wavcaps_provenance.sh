#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MECAT_MANIFEST="${DATA_ROOT}/mecat_caption_be4a24c3/manifests/mecat_00a_test_manifest.jsonl"
WAVCAPS_ROOT="${DATA_ROOT}/wavcaps_0930ec11/source"
AUDIT_ROOT="${DATA_ROOT}/wavcaps_0930ec11/audits"
RUN_ID="data11_mecat_wavcaps_provenance_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
OVERLAP_OUTPUT="${AUDIT_ROOT}/mecat_00a_test_wavcaps_audioset_source_video_matches.jsonl"

mkdir -p "${RUN_DIR}" "${AUDIT_ROOT}"

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
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] MECAT manifest: ${MECAT_MANIFEST}"
echo "[INFO] WavCaps metadata root: ${WAVCAPS_ROOT}"
echo "[INFO] GPU disabled"
echo "[INFO] Metadata-only source-video audit; no audio is read or modified"

set +e
python scripts/audit_mecat_wavcaps_provenance.py \
  --mecat-manifest "${MECAT_MANIFEST}" \
  --wavcaps-root "${WAVCAPS_ROOT}" \
  --overlap-output "${OVERLAP_OUTPUT}" \
  --statistics-output "${RUN_DIR}/data_statistics.json" \
  --expected-mecat-examples 848
AUDIT_EXIT_CODE=$?
set -e
printf '%s\n' "${AUDIT_EXIT_CODE}" > "${RUN_DIR}/audit_exit_code.txt"

if [[ "${AUDIT_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] DATA-11 provenance audit failed; evidence was preserved." >&2
  echo "[ERROR] Audit artifacts: ${RUN_DIR}" >&2
  exit "${AUDIT_EXIT_CODE}"
fi

sha256sum "${OVERLAP_OUTPUT}" > "${RUN_DIR}/overlap_artifact_sha256.txt"
echo "[INFO] DATA-11 MECAT/WavCaps source-video provenance audit completed"
echo "[WARN] Source-video matches are candidates, not proof of temporal/audio overlap"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
echo "[INFO] Candidate artifact: ${OVERLAP_OUTPUT}"
