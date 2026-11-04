#!/usr/bin/env bash
# Run the project Conda Python with Ubuntu's simulator system libraries.
set -euo pipefail
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "$script_path")/../.." && pwd)"
user_root="$(cd "$project_root/../.." && pwd)"
environment_root="${OU_ENV_ROOT:-$user_root/envs/safetymimic}"
image="${SAFETYMIMIC_RUNTIME_IMAGE:-$project_root/.cache/containers/ubuntu24-safetymimic.sif}"
apptainer="${APPTAINER:-$project_root/.cache/apptainer-public-1.5.1/bin/apptainer}"
[[ -x "$apptainer" ]] || { echo "Missing Apptainer: $apptainer; run scripts/hpc/setup_env.sh" >&2; exit 2; }
[[ -x "$environment_root/bin/python" && -f "$image" ]] || { echo 'Missing Conda Python or Ubuntu runtime; run scripts/hpc/setup_env.sh' >&2; exit 2; }
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$project_root/.cache/apptainer}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$user_root/.cache/uv-ubuntu24}"
export PYTHONNOUSERSITE=1
if [[ -n "${SLURM_JOB_ID:-}" && -z "${SAFETYMIMIC_TASK_CPUS:-}" ]]; then
  export SAFETYMIMIC_TASK_CPUS="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)"
fi
exec "$apptainer" exec --nv --bind "$user_root:$user_root" --pwd "$PWD" \
  --env "ISAACLAB_PATH=$project_root/vendor/supermimic/IsaacLab" \
  --env "PYTHONPATH=$project_root/src:$project_root/vendor/supermimic" \
  "$image" "$environment_root/bin/python" "$@"
