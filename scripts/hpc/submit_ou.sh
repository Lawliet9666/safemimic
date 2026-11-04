#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OU_PARTITION:?Set OU_PARTITION to the actual OU Blackwell partition; its name is not assumed}"
mkdir -p "$project_root/outputs/logs"
options=(--partition="$OU_PARTITION" --chdir="$project_root"
  --output="$project_root/outputs/logs/%x_%j.log")
if [[ "$OU_PARTITION" == gpu6000 || "$OU_PARTITION" == gpu6000-short ]]; then
  options+=(--gres="${OU_GRES:-gpu:6000:2}")
elif [[ -n "${OU_GRES:-}" ]]; then
  options+=(--gres="$OU_GRES")
fi
[[ -z "${OU_ACCOUNT:-}" ]] || options+=(--account="$OU_ACCOUNT")
[[ -z "${OU_CONSTRAINT:-}" ]] || options+=(--constraint="$OU_CONSTRAINT")
cd "$project_root"
# Additional sbatch options, e.g. --time, can be passed to this wrapper.
exec sbatch "${options[@]}" "$@" "$project_root/scripts/hpc/train_ou.sbatch"
