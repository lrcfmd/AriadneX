#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

target_env="${ARIADNEX_ENV:-ariadnex}"
conda_exe="${CONDA_EXE:-conda}"
active_env="${CONDA_DEFAULT_ENV:-}"
if [[ -z "$active_env" && -n "${CONDA_PREFIX:-}" ]]; then
  active_env="$(basename -- "$CONDA_PREFIX")"
fi

if [[ "$active_env" == "$target_env" ]]; then
  exec python -m ariadnex.cli "$@"
fi

if ! command -v "$conda_exe" >/dev/null 2>&1; then
  echo "Conda executable not found: $conda_exe" >&2
  echo "Activate $target_env or set CONDA_EXE before running workflow." >&2
  exit 127
fi

exec "$conda_exe" run --no-capture-output -n "$target_env" python -m ariadnex.cli "$@"
