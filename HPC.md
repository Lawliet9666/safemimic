# OU HPC training

The local wrapper remains restricted to GPU 0, CPUs 0–5 and a six-core quota.
The Slurm profile is a separate, explicit execution mode and requires an actual
Slurm allocation. It partitions the assigned CPU set between workers and uses
the two logical CUDA devices exposed by Slurm. It does not assume physical CPU
IDs or override `CUDA_VISIBLE_DEVICES`.

## Environment and submission

Copy the repository, pinned vendor/assets, pretrained actor and frozen world
model onto the shared filesystem. The ignored assets are not included by Git.
Prefer an Apptainer SIF containing the existing tested Docker environment
(`safetymimic:46d6cc3-cu12`, PyTorch 2.11/CUDA 12.8, qpth and the pinned
Isaac Lab/Newton/MuJoCo-Warp stack). `SAFETYMIMIC_SIF` must point to a SIF that
has already been built and checked. This script does not require a Docker daemon
on compute nodes. Alternatively activate a compatible micromamba environment
before submission and set `OU_PYTHON` to its absolute Python executable; no
project `.venv` is created. Installing only pyproject dependencies is insufficient
to recreate the simulation stack.

```bash
cd /path/on/ou/safetymimic
export SAFETYMIMIC_SIF=/path/on/ou/safetymimic.sif
export OU_PARTITION=ACTUAL_BLACKWELL_PARTITION
# Optional, if required by your allocation:
# export OU_ACCOUNT=your_account
# export OU_CONSTRAINT=actual_site_constraint
bash scripts/hpc/submit_ou.sh
```

Before the first training job, submit `PREFLIGHT_ONLY=1 bash scripts/hpc/submit_ou.sh
--time=00:15:00`. It checks actual CUDA/NCCL gradient averaging and each rank's
CPU assignment, and writes `outputs/results/ou_preflight_JOBID.json`. It does
not test the simulator. Then use `UPDATES=10` for a bounded simulation profile.

The actual Blackwell queue and node CPU count need confirmation from OU/OSCER.
No Blackwell partition name is invented or hardcoded. OU's public
[partition page](https://www.ou.edu/oscer/using-the-cluster/partitions) and
[submission guide](https://www.ou.edu/oscer/using-the-cluster/submitting-jobs)
describe the existing queues and Slurm submission. The new node may have
additional site requirements. The scripts request generic `--gres=gpu:2`, so
the chosen queue/constraint must actually select the intended hardware.

`submit_ou.sh` accepts extra sbatch arguments, for example `--time=08:00:00`.
Run the shell syntax tests (`pytest tests/test_hpc_launch.py`) before submitting.
Do not invoke the local Docker wrapper inside this job: it deliberately exposes
only one GPU. Apptainer uses `--nv` and preserves the scheduler's GPU assignment.

## Initial configuration

One node, **one Slurm task**, **two torchrun workers**:

| Setting | Default |
|---|---:|
| GPUs | 2 |
| CPUs / host memory / time request | 32 / 128 GB / 24 hours |
| Threads per worker | 16 |
| Simulator environments per GPU | 64 |
| Microbatch per GPU | 8 |
| Accumulation steps | 2 |
| Global actor batch | 8 × 2 × 2 = 32 |
| Imagined horizon | 32 control steps |
| Original encoder/decoder learning rate | 1e-5 |
| Ball encoder learning rate | 5e-5 |
| Critic learning rate | 1e-4 |
| Warmup / replay capacity | 256 / 1,000 physical control steps per worker |
| Actor/critic gradient clipping | norm 1 |
| Full checkpoint interval | 100 updates, plus requested milestones and final |

These are starting settings, not demonstrated optimal hyperparameters. First
submit a short profile with `UPDATES=10`. If QP/BPTT memory is excessive, reduce
`BATCH_SIZE=4 ACCUMULATION_STEPS=4` while preserving the global batch of 32.
Track QP iteration counts, failure types, latency and memory before a long run.
The frozen world model is unchanged. The actor still uses one QP-guided rollout
and exactly the correction/slack auxiliary losses. Deployment still omits QP.

Fresh-run overrides are environment variables:
`SEED`, `METHOD`, `NUM_ENVS`, `BATCH_SIZE`, `ACCUMULATION_STEPS`, `HORIZON`,
`ACTOR_LR`, `BALL_LR`, `CRITIC_LR`, `WARMUP_STEPS`, `REPLAY_STEPS`, `UPDATES`,
`CPU_THREADS`, and `RUN_DIR`. Defaults use `outputs/runs/ou_JOBID`.
Each rank has its own simulation, exploration stream, replay and compilation
cache. Completed actor gradients are averaged **after** all accumulated
backward passes and **before** clipping/Adam. This avoids distributed reducer
hooks inside the QP's nested Jacobian calculations. The critic uses globally
gathered normalization data and synchronized gradients as well.

## Resume and preemption

```bash
export RUN_DIR=outputs/runs/ou_12345
export RESUME=outputs/runs/ou_12345/checkpoints/latest.json
export UPDATES=3000
bash scripts/hpc/submit_ou.sh
```

`UPDATES` is the **total target**, not an additional number. Resume requires
the same world size, simulator batch, objective and optimizer settings. Semantic
settings are restored from the snapshot; conflicting explicit command-line
overrides are rejected. Learning rates and Adam moments are restored, not reset.
Single-GPU checkpoints resume on one GPU; two-GPU checkpoints resume on two.
Resume selects the published latest snapshot; rolling back in the same directory
is rejected to prevent duplicate update numbers and ambiguous metric history.

The snapshot contains actor, critic, both optimizers, counters, replay, RNG state,
episode counters and observation histories. Each rank writes its own shard.
Rank 0 publishes an immutable manifest and then atomically advances
`checkpoints/latest.json` only after all shards have completed. Actor-only
`policy.pt` remains the deployment artifact; it is not a resume checkpoint.

The physics solver's hidden contact state is not serialized. On resume,
unfinished physical episodes are explicitly marked **censored** and reset;
stored replay and optimizer/RNG state are retained. This is training-state
resume, not bitwise continuation of an interrupted contact simulation.

Ten minutes before the time limit, Slurm signals the batch shell, which writes
a job-specific stop marker. Workers finish the current optimizer update, save
a complete snapshot, and write `STOPPED.json`. Signals delivered directly to a
worker also request graceful stopping. A hard kill before the snapshot finishes
leaves the preceding published checkpoint intact. Checkpoint latency and the
longest update must fit inside the warning interval; increase the Slurm signal
lead time for unusually expensive profiles. `STATUS.json` is authoritative for
the current invocation; prior completion/error artifacts are retained as history.

## Local verification

```bash
bash scripts/container.sh python -m pytest -q
bash scripts/container.sh python scripts/train.py \
  --num_envs 2 --batch_size 1 --horizon 2 --updates 2 \
  --gradient_accumulation_steps 2 --warmup_steps 0 --replay_steps 8 \
  --critic_epochs 1 --critic_minibatches 2 \
  --checkpoint_updates 1,2 --checkpoint_every 0 \
  --output outputs/runs/local_hpc_smoke
bash scripts/container.sh python scripts/train.py \
  --resume outputs/runs/local_hpc_smoke/checkpoints/latest.json --updates 3
```

All these commands stay on local GPU 0 and at most six cores. The two-process
distributed tests use CPU/Gloo, not local GPU 1. Actual two-GPU NCCL, Blackwell
memory/throughput, Apptainer and OU queue execution require verification on the
allocated node; local tests do not establish those hardware results.
