#!/usr/bin/env bash
# Bounded pilot; continuation requires reviewing the paired evaluation gate.
#SBATCH --job-name=sm-avoid-pilot
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
git rev-parse HEAD
export OU_PYTHON="${OU_PYTHON:-$PWD/scripts/hpc/python.sh}"
export CPU_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export OMP_THREAD_LIMIT=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
export METHOD=filter AVOIDANCE_WEIGHT=10 UPDATES=100 NUM_ENVS=64 BATCH_SIZE=8 ACCUMULATION_STEPS=2
export HORIZON=32 WARMUP_STEPS=48 REPLAY_STEPS=1000 SEED=0
export RUN_DIR="$PWD/outputs/runs/filter_avoidance_${SLURM_JOB_ID}"
export STOP_FILE="$PWD/outputs/logs/stop_avoidance_${SLURM_JOB_ID}"
unset RESUME
mkdir -p outputs/logs
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
echo 'STAGE: 100-update avoidance pilot'
run_step bash scripts/hpc/launch.sh
[[ ! -f "$STOP_FILE" ]] || exit 0
for phase in before after; do
  checkpoint="$RUN_DIR/policy.pt"
  if [[ "$phase" == before ]]; then checkpoint="$RUN_DIR/policy_000000.pt"; fi
  echo "STAGE: fixed-throw evaluation $phase"
  run_step "$OU_PYTHON" scripts/evaluate_safety.py --runtime slurm --cpu_threads 16 \
    --manifest outputs/manifests/experiment_20261104/development.json \
    --checkpoint "$checkpoint" --output "$RUN_DIR/eval_$phase" \
    --num_envs 8 --limit 8 --label "filter_avoidance_$phase"
  [[ ! -f "$STOP_FILE" ]] || exit 0
  echo "STAGE: unfiltered standing $phase"
  run_step "$OU_PYTHON" tests/integration/validate_starting_policy.py --runtime slurm --cpu_threads 16 \
    --checkpoint "$checkpoint" --steps 300 --num_envs 8 --report_only \
    --output "$RUN_DIR/standing_$phase.json"
  [[ ! -f "$STOP_FILE" ]] || exit 0
done
run_step "$OU_PYTHON" tests/integration/gate_filter_pilot.py --run "$RUN_DIR" \
  --before "$RUN_DIR/eval_before" --after "$RUN_DIR/eval_after" \
  --standing_before "$RUN_DIR/standing_before.json" --standing_after "$RUN_DIR/standing_after.json" \
  --output "outputs/analysis/filter_avoidance_pilot_${SLURM_JOB_ID}"
echo "PILOT_COMPLETE: outputs/analysis/filter_avoidance_pilot_${SLURM_JOB_ID}.md"
