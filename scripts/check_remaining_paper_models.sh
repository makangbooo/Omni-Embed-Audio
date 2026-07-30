#!/usr/bin/env bash
set -uo pipefail

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
COMPLETE=0
MISSING=0

check_source() {
  local id="$1" path="$2" revision="$3"
  local marker="${path}/.source_revision"
  if [[ -f "${marker}" ]] && [[ "$(cat "${marker}")" == "${revision}" ]]; then
    echo "ASSET=${id} STATUS=complete BYTES=$(du -sb "${path}" | awk '{print $1}') PATH=${path}"
    COMPLETE=$((COMPLETE + 1))
  else
    echo "ASSET=${id} STATUS=missing PATH=${path}"
    MISSING=$((MISSING + 1))
  fi
}

check_file() {
  local id="$1" path="$2"
  if [[ -s "${path}" ]]; then
    echo "ASSET=${id} STATUS=complete BYTES=$(stat -c %s "${path}") PATH=${path}"
    COMPLETE=$((COMPLETE + 1))
  else
    echo "ASSET=${id} STATUS=missing PATH=${path}"
    MISSING=$((MISSING + 1))
  fi
}

check_snapshot() {
  local path="${MODEL_ROOT}/bge-large-en-v1.5"
  local revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09"
  if [[ -s "${path}/model.safetensors" ]] \
     && [[ -f "${path}/.source_revision" ]] \
     && [[ "$(cat "${path}/.source_revision")" == "${revision}" ]]; then
    echo "ASSET=bge_snapshot STATUS=complete BYTES=$(du -sb "${path}" | awk '{print $1}') PATH=${path}"
    COMPLETE=$((COMPLETE + 1))
  else
    echo "ASSET=bge_snapshot STATUS=missing PATH=${path}"
    MISSING=$((MISSING + 1))
  fi
}

check_source robust_source "${MODEL_ROOT}/robust-clap/source" \
  d08d0e3c545fa22df0930fc0d090741aaa9e2cc1
check_source mga_source "${MODEL_ROOT}/mga-clap/source" \
  48ca5a5cd22cd34427e118bd8cf332090ec54770
check_file mga_checkpoint \
  "${MODEL_ROOT}/mga-clap/pretrained_models/models/model.pt"
check_source m2d_source "${MODEL_ROOT}/m2d-clap/source" \
  3d0c4de9447c404a8d3f9f37e04f53bc902e09b3
check_file m2d_checkpoint \
  "${MODEL_ROOT}/m2d-clap/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"
check_snapshot

echo "ROBUST_CHECKPOINT_STATUS=blocked_not_published"
echo "COMPLETE=${COMPLETE}/6"
echo "MISSING=${MISSING}/6"
if [[ "${MISSING}" -eq 0 ]]; then
  echo "FINAL_CHECK_RC=0"
  exit 0
fi
echo "FINAL_CHECK_RC=1"
exit 1
