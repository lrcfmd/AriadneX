#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
repository="lrcfmd/AriadneX"
datasets=(controlled_leave_one_out open_world_phase_discovery)
model_only=false
refresh_examples=false
data_release="example-data-v5"

case "${1:-}" in
  --model-only) model_only=true ;;
  --refresh-examples) refresh_examples=true ;;
  "") ;;
  -h|--help)
    echo "Usage: $0 [--model-only | --refresh-examples]"
    echo "Download the model and experimental datasets. Use --model-only for your own data."
    exit 0
    ;;
  *)
    echo "Unknown argument: $1. Use --help for usage." >&2
    exit 2
    ;;
esac
if [[ $# -gt 1 ]]; then
  echo "Too many arguments. Use --help for usage." >&2
  exit 2
fi

has_csv_rows() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  awk 'NR > 1 && $0 !~ /^[[:space:]]*$/ { found = 1; exit } END { exit(found ? 0 : 1) }' "$path"
}

has_nonempty_file() {
  local directory="$1"
  local pattern="$2"
  [[ -d "$directory" ]] || return 1
  find "$directory" -type f -name "$pattern" -size +0c -print -quit 2>/dev/null | grep -q .
}

example_data_ready() {
  local dataset dataset_root file
  for dataset in "${datasets[@]}"; do
    dataset_root="$repo_root/data/$dataset"
    for file in samples.csv ground_truth.csv basis_order.csv; do
      has_csv_rows "$dataset_root/$file" || return 1
    done
    [[ -s "$dataset_root/experimental.instprm" ]] || return 1
    has_nonempty_file "$dataset_root/mixed_patterns" '*' || return 1
  done
  [[ -s "$repo_root/data/open_world_phase_discovery/candidate_library.instprm" ]]
}

need_data=false
if ! example_data_ready || "$refresh_examples"; then
  need_data=true
elif [[ "$(cat "$repo_root/data/.example-data-version" 2>/dev/null || true)" != "$data_release" ]] && ! "$model_only"; then
  echo "The example data needs updating. Run $0 --refresh-examples; existing examples will be backed up." >&2
  exit 1
fi
if "$model_only"; then
  need_data=false
fi

need_model=false
if [[ ! -s "$repo_root/models/last.ckpt" ]]; then
  need_model=true
fi

if "$need_model" || "$need_data"; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "Install the GitHub CLI (gh), sign in with 'gh auth login', and rerun this command." >&2
    exit 1
  fi
  download_dir="$(mktemp -d)"
  trap 'rm -rf -- "$download_dir"' EXIT

  if "$need_model"; then
    echo "Downloading the PXRD-patch transformer model..."
    gh release download checkpoint-v1 --repo "$repository" --pattern last.ckpt --dir "$download_dir"
    mkdir -p "$repo_root/models"
    mv "$download_dir/last.ckpt" "$repo_root/models/last.ckpt"
  fi
  if "$need_data"; then
    echo "Downloading the experimental PXRD datasets..."
    gh release download "$data_release" --repo "$repository" --pattern example-data.tar.gz --dir "$download_dir"
    tar -xzf "$download_dir/example-data.tar.gz" -C "$download_dir"
    mkdir -p "$repo_root/data"
    if "$refresh_examples"; then
      backup_dir="$(mktemp -d "$repo_root/data/example-backup-XXXXXXXX")"
      for dataset in "${datasets[@]}"; do
        if [[ -e "$repo_root/data/$dataset" ]]; then
          mv "$repo_root/data/$dataset" "$backup_dir/$dataset"
        fi
      done
      echo "Previous examples saved in $backup_dir"
    fi
    cp -Rn "$download_dir/data/." "$repo_root/data/"
    cp "$download_dir/data/.example-data-version" "$repo_root/data/.example-data-version"
  fi
fi

if "$model_only"; then
  echo "Model is ready. Follow 'Your own data' in README.md to add your inputs."
else
  if ! example_data_ready; then
    echo "Example data is incomplete. Run $0 --refresh-examples to download it again." >&2
    exit 1
  fi
  echo "Model and experimental datasets are ready. Existing files have been kept."
  echo "The paper examples need your local CIFs. Follow 'Prepare and run a paper example' in README.md."
fi
echo "Next: follow README.md to prepare your inputs and run a config."
