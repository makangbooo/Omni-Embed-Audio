#!/usr/bin/env bash

# Resolve the Conda installation that initialized the calling shell.  This is
# intentionally independent of PATH because tmux child shells can retain an
# active environment while resolving the `conda` executable from another base.
resolve_conda_executable() {
  local candidate="${CONDA_EXE:-}"
  local prefix="${CONDA_PREFIX:-}"
  local base_candidate=""

  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  if [[ -n "${prefix}" ]]; then
    if [[ -x "${prefix}/bin/conda" ]]; then
      printf '%s\n' "${prefix}/bin/conda"
      return 0
    fi
    if [[ "$(basename "$(dirname "${prefix}")")" == "envs" ]]; then
      base_candidate="$(dirname "$(dirname "${prefix}")")/bin/conda"
      if [[ -x "${base_candidate}" ]]; then
        printf '%s\n' "${base_candidate}"
        return 0
      fi
    fi
  fi

  candidate="$(command -v conda 2>/dev/null || true)"
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  echo "[ERROR] Unable to resolve a Conda executable." >&2
  return 1
}

resolve_conda_base() {
  local conda_executable
  conda_executable="$(resolve_conda_executable)"
  "${conda_executable}" info --base
}
