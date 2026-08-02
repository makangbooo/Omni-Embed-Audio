#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <weighted-variant> <completed-positive-uiq-result-dir>" >&2
  exit 2
fi

exec bash "${ROOT_DIR}/scripts/run_mecat_positive_uiq.sh" \
  "$1" table2_table3 "$2"
