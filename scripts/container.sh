#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec docker run --rm --gpus '"device=0"' --shm-size=8g --memory=96g \
  --cpuset-cpus 0-5 --cpus 6 \
  --user "$(id -u):$(id -g)" \
  --env PYTHONUNBUFFERED=1 \
  --env USER=safetymimic --env LOGNAME=safetymimic \
  --env XDG_CACHE_HOME=/workspace/safetymimic/.cache \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env OMP_NUM_THREADS=6 --env MKL_NUM_THREADS=6 --env OPENBLAS_NUM_THREADS=6 \
  --env NUMEXPR_NUM_THREADS=6 --env OMP_THREAD_LIMIT=6 \
  --volume "$task_root:/workspace/safetymimic" \
  safetymimic:46d6cc3-cu12 "$@"
