#!/usr/bin/env bash
# Bounded end-to-end comparison; no full fine-tuning run is submitted here.
#SBATCH --job-name=sm-filter-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6000:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@900
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
git diff --quiet && git diff --cached --quiet
export OU_PYTHON="${OU_PYTHON:-$PWD/scripts/hpc/python.sh}"
export CPU_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export OMP_THREAD_LIMIT=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
export UPDATES="${COMPARISON_UPDATES:-2}" NUM_ENVS=64 BATCH_SIZE=8 ACCUMULATION_STEPS=2
export HORIZON=32 WARMUP_STEPS=48 REPLAY_STEPS=1000 SEED=0
unset RESUME
root_run="outputs/runs/filter_compare_${SLURM_JOB_ID}"
export STOP_FILE="$PWD/outputs/logs/stop_compare_${SLURM_JOB_ID}"
mkdir -p outputs/logs "$root_run"
git rev-parse HEAD > "$root_run/source_commit.txt"
trap 'touch "$STOP_FILE"' USR1 TERM INT

run_step() {
  srun --ntasks=1 --cpu-bind=cores bash -c \
    'export SAFETYMIMIC_TASK_CPUS="$(awk "/^Cpus_allowed_list:/ {print \$2}" /proc/self/status)"; exec "$@"' \
    bash "$@" &
  child=$!
  while true; do
    if wait "$child"; then return 0; else status=$?; fi
    kill -0 "$child" 2>/dev/null || return "$status"
  done
}

echo 'STAGE: unit tests'
run_step "$OU_PYTHON" -m pytest -q
echo 'STAGE: external-filter integration smoke'
export METHOD=filter RUN_DIR="$PWD/$root_run/smoke"
UPDATES=1 NUM_ENVS=2 BATCH_SIZE=1 ACCUMULATION_STEPS=1 HORIZON=2 WARMUP_STEPS=0 \
  run_step bash scripts/hpc/launch.sh
[[ ! -f "$STOP_FILE" ]] || exit 0

for method in supermimic filter safetymimic; do
  export METHOD="$method" RUN_DIR="$PWD/$root_run/$method"
  echo "STAGE: matched training $method"
  run_step bash scripts/hpc/launch.sh
  [[ ! -f "$STOP_FILE" ]] || exit 0
  echo "STAGE: fixed-throw evaluation $method"
  run_step "$OU_PYTHON" scripts/evaluate_safety.py --runtime slurm --cpu_threads 16 \
    --manifest outputs/manifests/experiment_20261104/development.json \
    --checkpoint "$root_run/$method/policy.pt" --output "$root_run/eval_$method" \
    --num_envs 8 --limit 8 --label "$method"
  if [[ "$method" == filter ]]; then
    echo 'STAGE: single-environment filter latency'
    run_step "$OU_PYTHON" scripts/evaluate_safety.py --runtime slurm --cpu_threads 16 \
      --manifest outputs/manifests/experiment_20261104/development.json \
      --checkpoint "$root_run/filter/policy.pt" --output "$root_run/eval_filter_batch1" \
      --num_envs 1 --limit 1 --label filter_batch1
  fi
  [[ ! -f "$STOP_FILE" ]] || exit 0
done
run_step "$OU_PYTHON" tests/integration/summarize_filter_comparison.py --root "$root_run" \
  --output "outputs/analysis/filter_comparison_${SLURM_JOB_ID}"
echo "COMPARISON_COMPLETE: outputs/analysis/filter_comparison_${SLURM_JOB_ID}.md"
