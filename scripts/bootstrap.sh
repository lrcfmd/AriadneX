#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
conda_exe="${CONDA_EXE:-conda}"
dry_run=false

for argument in "$@"; do
  case "$argument" in
    --dry-run) dry_run=true ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $argument" >&2
      exit 2
      ;;
  esac
done

if ! command -v "$conda_exe" >/dev/null 2>&1; then
  echo "Conda executable not found: $conda_exe" >&2
  echo "Set CONDA_EXE to the full path of your conda executable and retry." >&2
  exit 2
fi

create_or_update() {
  local environment_file="$1"
  local env_name
  local dry_root
  env_name="$(awk -F': *' '$1 == "name" {print $2; exit}' "$environment_file")"
  if [[ -z "$env_name" ]]; then
    echo "No environment name in $environment_file" >&2
    exit 2
  fi

  if "$dry_run"; then
    dry_root="$(mktemp -d)"
    echo "Checking environment solve: $env_name"
    "$conda_exe" env create --prefix "$dry_root/$env_name" --file "$environment_file" --dry-run
    rmdir "$dry_root"
    return
  fi

  if "$conda_exe" env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "Updating existing environment: $env_name"
    "$conda_exe" env update --name "$env_name" --file "$environment_file"
  else
    echo "Creating environment: $env_name"
    "$conda_exe" env create --file "$environment_file"
  fi
}

environment_file="$repo_root/envs/environment.yml"
create_or_update "$environment_file"

if "$dry_run"; then
  echo "Environment dry-run completed."
else
  env_name="$(awk -F': *' '$1 == "name" {print $2; exit}' "$environment_file")"
  "$conda_exe" run -n "$env_name" python -m pip install -e "$repo_root"
  "$conda_exe" run -n "$env_name" python -c '
from pathlib import Path
import site
import subprocess
import sys

gsasii = Path(sys.prefix) / "GSAS-II" / "GSASII"
if not (gsasii / "GSASIIscriptable.py").is_file():
    raise SystemExit(f"GSAS-II installation not found: {gsasii}")
# gsas2pkg downloads the latest source; use the revision used by our experiments.
gsasii_revision = "1a0aeec6c61b54f68e6812704730a471187ac393"
subprocess.run(
    ["git", "-C", str(gsasii.parent), "fetch", "--quiet", "origin", gsasii_revision],
    check=True,
)
subprocess.run(
    ["git", "-C", str(gsasii.parent), "checkout", "--quiet", "--detach", gsasii_revision],
    check=True,
)
Path(site.getsitepackages()[0], "ariadnex_gsasii.pth").write_text(f"{gsasii}\n", encoding="utf-8")
'
  echo "Environment ready. Set up your Gurobi licence as described in README.md."
  echo "Next: prepare your local CIFs with ./workflow prepare-example --help. See README.md."
fi
