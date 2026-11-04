#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${STOP_FILE:?Set STOP_FILE}"
project_root="$PWD"
export PYTHONPATH="$project_root/src:$project_root/vendor/supermimic${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$project_root/.cache"
export MUJOCO_GL=egl MPLCONFIGDIR="${TMPDIR:-/tmp}/safetymimic-mpl-${SLURM_JOB_ID}"
python_executable="${OU_PYTHON:-python}"
"$python_executable" -c 'import torch; assert torch.cuda.device_count() == 2, "Expected exactly two Slurm-assigned visible GPUs"; print(torch.__version__, torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(2)], flush=True)'
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exec "$python_executable" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
    tests/integration/preflight.py
fi
training=(--runtime slurm --cpu_threads "${CPU_THREADS:-16}"
  --updates "${UPDATES:-3000}" --output "$RUN_DIR" --stop_file "$STOP_FILE")
if [[ -n "${RESUME:-}" ]]; then
  training+=(--resume "$RESUME")
else
  training+=(--num_envs "${NUM_ENVS:-64}" --batch_size "${BATCH_SIZE:-8}"
    --gradient_accumulation_steps "${ACCUMULATION_STEPS:-2}" --horizon "${HORIZON:-32}"
    --actor_lr "${ACTOR_LR:-1e-5}" --ball_lr "${BALL_LR:-5e-5}" --critic_lr "${CRITIC_LR:-1e-4}"
    --warmup_steps "${WARMUP_STEPS:-256}" --replay_steps "${REPLAY_STEPS:-1000}"
    --seed "${SEED:-0}" --method "${METHOD:-safetymimic}")
fi
# CUDA_VISIBLE_DEVICES is supplied by Slurm and deliberately preserved.
exec "$python_executable" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/hpc/worker.py "${training[@]}"
