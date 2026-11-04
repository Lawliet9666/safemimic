#!/usr/bin/env bash
#SBATCH --job-name=safetymimic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?}"
project_root="$PWD"
export CPU_THREADS="${CPU_THREADS:-$((SLURM_CPUS_PER_TASK / 2))}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
export OMP_THREAD_LIMIT="$CPU_THREADS" PYTHONUNBUFFERED=1
export RUN_DIR="${RUN_DIR:-outputs/runs/ou_${SLURM_JOB_ID}}"
if [[ "$RUN_DIR" != /* ]]; then RUN_DIR="$project_root/$RUN_DIR"; fi
case "$RUN_DIR/" in "$project_root/outputs/runs/"*) ;; *) echo 'RUN_DIR must be inside outputs/runs' >&2; exit 2;; esac
# Do not create RUN_DIR yet: the trainer rejects nonempty fresh run directories.
mkdir -p "$project_root/outputs/logs" "$project_root/.cache"
stop_file="$project_root/outputs/logs/stop_${SLURM_JOB_ID}"
export STOP_FILE="$stop_file"
request_checkpoint() { touch "$stop_file"; }
trap request_checkpoint USR1 TERM INT
if [[ -n "${SAFETYMIMIC_SIF:-}" ]]; then
  [[ -f "$SAFETYMIMIC_SIF" ]] || { echo 'Missing SAFETYMIMIC_SIF' >&2; exit 2; }
  launcher=(apptainer exec --nv --bind "$project_root:$project_root" --pwd "$project_root"
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?Slurm must expose its assigned GPUs}"
    --env "PYTHONPATH=$project_root/src:$project_root/vendor/supermimic"
    "$SAFETYMIMIC_SIF" bash scripts/hpc/launch.sh)
else
  # A compatible Python environment must already be activated in this mode.
  launcher=(bash scripts/hpc/launch.sh)
fi
# ONE Slurm task launches TWO torchrun workers; never multiply both counts.
srun --ntasks=1 --cpu-bind=cores bash -c \
  'export SAFETYMIMIC_TASK_CPUS="$(awk "/^Cpus_allowed_list:/ {print \$2}" /proc/self/status)"; exec "$@"' \
  bash "${launcher[@]}" &
child=$!
status=0
# A trap can interrupt wait while srun is still running. Continue waiting so
# the workers can finish their current update and publish a complete snapshot.
while true; do
  if wait "$child"; then status=0; break; else status=$?; fi
  kill -0 "$child" 2>/dev/null || break
done
exit "$status"
