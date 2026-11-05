#!/usr/bin/env bash
# Shared adaptive-gain verification, then a bounded external-filter-only pilot.
#SBATCH --job-name=sm-gain-filter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
git diff --quiet && git diff --cached --quiet
git rev-parse HEAD
export CPU_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export OMP_THREAD_LIMIT=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$PWD/.cache" MUJOCO_GL=egl
python_executable="$PWD/scripts/hpc/python.sh"
run="outputs/runs/gain_filter_${SLURM_JOB_ID}"
run_step() {
  srun --ntasks=1 --cpu-bind=cores bash -c \
    'export SAFETYMIMIC_TASK_CPUS="$(awk "/^Cpus_allowed_list:/ {print \$2}" /proc/self/status)"; exec "$@"' \
    bash "$@"
}
echo 'STAGE: unit tests'
run_step "$python_executable" -m pytest -q
echo 'STAGE: real-model gain gradient and forward equivalence'
run_step "$python_executable" tests/integration/validate_layer.py --runtime slurm --cpu_threads 16 \
  --adaptive_gain --gain_finite_difference --output "outputs/results/gain_validation_${SLURM_JOB_ID}.json"
echo 'STAGE: gain-only external-filter training'
run_step "$python_executable" scripts/train_gain_filter.py --runtime slurm --cpu_threads 16 \
  --manifest outputs/manifests/experiment_20261104/development.json --output "$run" \
  --num_envs 8 --rounds 4 --updates_per_round 25
for phase in before after; do
  checkpoint="$run/policy.pt"
  if [[ "$phase" == before ]]; then checkpoint="$run/policy_000000.pt"; fi
  echo "STAGE: fixed-throw evaluation $phase"
  run_step "$python_executable" scripts/evaluate_safety.py --runtime slurm --cpu_threads 16 \
    --manifest outputs/manifests/experiment_20261104/development.json --checkpoint "$checkpoint" \
    --output "$run/eval_$phase" --num_envs 8 --limit 8 --label "adaptive_gain_$phase"
  echo "STAGE: unfiltered standing $phase"
  run_step "$python_executable" tests/integration/validate_starting_policy.py --runtime slurm --cpu_threads 16 \
    --checkpoint "$checkpoint" --steps 300 --num_envs 8 --report_only --output "$run/standing_$phase.json"
done
run_step "$python_executable" tests/integration/summarize_gain_pilot.py --run "$run" \
  --output "outputs/analysis/gain_filter_pilot_${SLURM_JOB_ID}"
echo "GAIN_PILOT_COMPLETE: outputs/analysis/gain_filter_pilot_${SLURM_JOB_ID}.md"
