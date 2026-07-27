#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Usage:
  bash scripts/monitor_asrur_phase2.sh \
    --run-dir <absolute_run_directory> \
    [--pid <wrapper_pid>] \
    [--interval <seconds>] \
    [--once] \
    [--no-clear]

The default mode refreshes until the wrapper exits. Press Ctrl-C to detach from
the monitor; this does not stop the experiment.
EOF
}

RUN_DIR=""
WRAPPER_PID=""
INTERVAL_SECONDS=5
FOLLOW=1
CLEAR_SCREEN=1

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run-dir)
      [[ "$#" -ge 2 ]] || {
        echo "[ERROR] --run-dir requires a value." >&2
        usage >&2
        exit 2
      }
      RUN_DIR=$2
      shift 2
      ;;
    --pid)
      [[ "$#" -ge 2 ]] || {
        echo "[ERROR] --pid requires a value." >&2
        usage >&2
        exit 2
      }
      WRAPPER_PID=$2
      shift 2
      ;;
    --interval)
      [[ "$#" -ge 2 ]] || {
        echo "[ERROR] --interval requires a value." >&2
        usage >&2
        exit 2
      }
      INTERVAL_SECONDS=$2
      shift 2
      ;;
    --once)
      FOLLOW=0
      shift
      ;;
    --no-clear)
      CLEAR_SCREEN=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "[ERROR] --run-dir is required." >&2
  usage >&2
  exit 2
fi
if [[ ! "${INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --interval must be a positive integer." >&2
  exit 2
fi
if [[ -n "${WRAPPER_PID}" && ! "${WRAPPER_PID}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --pid must be a positive integer." >&2
  exit 2
fi

RUN_DIR="$(readlink -f "${RUN_DIR}")"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] Run directory does not exist: ${RUN_DIR}" >&2
  exit 3
fi

RUN_IDENTITY="${RUN_DIR}/run_identity.txt"
RESULT_ROOT=""
if [[ -f "${RUN_IDENTITY}" ]]; then
  RESULT_ROOT="$(
    awk -F= '$1 == "result_root" {
      sub(/^result_root=/, "")
      print
      exit
    }' "${RUN_IDENTITY}"
  )"
fi

latest_step_directory() {
  find "${RUN_DIR}/steps" \
    -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

render_snapshot() {
  local process_running="UNKNOWN"
  local wrapper_status="PENDING"
  local latest_step=""
  local progress_line=""

  if [[ "${FOLLOW}" -eq 1 && "${CLEAR_SCREEN}" -eq 1 && -t 1 ]]; then
    printf '\033[2J\033[H'
  fi

  echo "===== ASRUR PHASE-2 LIVE MONITOR ====="
  printf 'timestamp=%s\n' "$(date -Is)"
  printf 'run_dir=%s\n' "${RUN_DIR}"
  printf 'result_root=%s\n' "${RESULT_ROOT:-UNKNOWN}"
  printf 'refresh_seconds=%s\n' "${INTERVAL_SECONDS}"

  echo
  echo "===== PROCESS ====="
  if [[ -n "${WRAPPER_PID}" ]]; then
    if kill -0 "${WRAPPER_PID}" 2>/dev/null; then
      process_running="YES"
      ps -fp "${WRAPPER_PID}" || true
    else
      process_running="NO"
      echo "wrapper_pid=${WRAPPER_PID} is not running"
    fi
  else
    echo "wrapper_pid=NOT_PROVIDED"
  fi
  printf 'process_running=%s\n' "${process_running}"
  pgrep -af \
    '[r]un_asrur_phase2_frozen_retrieval.sh --execute|[g]enerate_asrur|[g]enerate_oea|[g]enerate_vanilla' \
    || true

  echo
  echo "===== GPU ====="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv 2>&1 || true
  else
    echo "nvidia-smi=UNAVAILABLE"
  fi

  echo
  echo "===== WRAPPER ====="
  if [[ -f "${RUN_DIR}/wrapper_exit_code.txt" ]]; then
    wrapper_status="$(tr -d '[:space:]' < "${RUN_DIR}/wrapper_exit_code.txt")"
  fi
  printf 'wrapper_exit_code=%s\n' "${wrapper_status}"

  echo
  echo "===== STEPS ====="
  for step_dir in "${RUN_DIR}"/steps/*; do
    [[ -d "${step_dir}" ]] || continue
    step_name="$(basename "${step_dir}")"
    if [[ -f "${step_dir}/exit_code.txt" ]]; then
      step_status="$(tr -d '[:space:]' < "${step_dir}/exit_code.txt")"
    else
      step_status="RUNNING_OR_PENDING"
    fi
    printf '%-32s %s\n' "${step_name}" "${step_status}"
  done

  latest_step="$(latest_step_directory)"
  echo
  echo "===== LATEST STEP ====="
  if [[ -n "${latest_step}" ]]; then
    printf 'latest_step=%s\n' "$(basename "${latest_step}")"
    if [[ -f "${latest_step}/command.sh" ]]; then
      printf 'command='
      tr '\n' ' ' < "${latest_step}/command.sh"
      echo
    fi
    progress_line="$(
      grep -h '^\[PROGRESS\]' "${latest_step}/stdout.log" 2>/dev/null \
        | tail -n 1
    )"
    printf 'latest_progress=%s\n' "${progress_line:-NONE}"
    echo "----- stdout tail -----"
    tail -n 8 "${latest_step}/stdout.log" 2>/dev/null || true
    echo "----- stderr tail -----"
    tail -n 8 "${latest_step}/stderr.log" 2>/dev/null || true
  else
    echo "latest_step=NONE"
  fi

  echo
  echo "===== METRICS ====="
  if [[ -n "${RESULT_ROOT}" && -d "${RESULT_ROOT}" ]]; then
    find "${RESULT_ROOT}" -type f -name metrics.json -print 2>/dev/null \
      | sort
  else
    echo "metrics=NONE"
  fi

  echo
  echo "===== COMPLETION ====="
  if [[ -f "${RUN_DIR}/completion_manifest.json" ]]; then
    cat "${RUN_DIR}/completion_manifest.json"
  else
    echo "completion_manifest=PENDING"
  fi

  if [[ "${wrapper_status}" != "PENDING" ]]; then
    return 10
  fi
  if [[ -n "${WRAPPER_PID}" && "${process_running}" == "NO" ]]; then
    return 11
  fi
  return 0
}

while true; do
  render_snapshot
  snapshot_status=$?
  if [[ "${FOLLOW}" -eq 0 ]]; then
    exit 0
  fi
  if [[ "${snapshot_status}" -eq 10 ]]; then
    echo
    echo "[INFO] Wrapper exit code is available; live monitoring finished."
    exit 0
  fi
  if [[ "${snapshot_status}" -eq 11 ]]; then
    echo
    echo "[WARNING] Wrapper PID exited without an exit-code artifact."
    exit 1
  fi
  sleep "${INTERVAL_SECONDS}"
done
