# Matilda HPC training

The local wrapper remains restricted to GPU 0, CPUs 0–5 and a six-core quota.
The Slurm profile is a separate, explicit execution mode and requires an actual
Slurm allocation. It partitions the assigned CPU set between workers and uses
the two logical CUDA devices exposed by Slurm. It does not assume physical CPU
IDs or override `CUDA_VISIBLE_DEVICES`.

## Environment and submission

Edit and commit code on the Mac. All simulator work, model fitting and training
run on the HPC. Push the reviewed revision, then use `git pull --ff-only` on the
HPC and verify the commit before running checks. Assets and environments are
ignored by Git and remain on the shared filesystem.

The pinned IsaacLab requires Python 3.12 and glibc >=2.35. Matilda's RHEL 8 host
has glibc 2.28, so a project Conda environment runs inside a small Ubuntu 24.04
Apptainer image. `scripts/hpc/python.sh` binds both the repository and environment;
use it for simulator commands. `conda activate` alone does not supply newer system
libraries. The environment defaults to `../../envs/safetymimic` relative to the
repository, or `OU_ENV_ROOT` when set.

For a fresh installation, load `miniconda3/current` and `git-gcc/2.31.1`, obtain a
CPU Slurm allocation (4 CPUs, 16 GB, two hours), and set `TMPDIR` to a writable
scratch directory. Then run:

```bash
mkdir -p "$TMPDIR"
srun --ntasks=1 bash scripts/hpc/setup_env.sh
```

The tracked setup script fetches `source_pins.json`, creates the Conda Python,
builds `scripts/hpc/ubuntu24.def`, installs the upstream packages and applies the
tested CUDA-12 and numerical-library adjustments. An existing environment can
be checked without reinstalling:

```bash
bash scripts/hpc/setup_env.sh --check
```

`tests/integration/check_env.py` verifies pinned versions, source revisions and imports.
It records the seven known metadata conflicts inherited from the pinned
IsaacLab overrides and qpth's NumPy cap, and fails on any other conflict. These
exceptions do not prove importer or simulator compatibility: asset conversion
and simulator execution are separate required checks. NumPy is fixed at 2.3.5;
the upstream 2.5 pin fails Numba's runtime import and Boost's declared range.

For the verified Conda/Ubuntu route:

```bash
export OU_PYTHON="$PWD/scripts/hpc/python.sh"
export OU_PARTITION=gpu6000-short
export OU_ACCOUNT=scale-lab
unset SAFETYMIMIC_SIF
PREFLIGHT_ONLY=1 bash scripts/hpc/submit_ou.sh --time=00:15:00
```

Matilda requires the typed GPU request `--gres=gpu:6000:2`; an untyped request
was routed to V100 GPUs, unsupported by this PyTorch build. The submission
wrapper adds the typed request for `gpu6000` and `gpu6000-short`. `OU_GRES` can
override it for another site. The short partition has a four-hour limit; use
`OU_PARTITION=gpu6000` for the default 24-hour training request.

`tests/integration/preflight.py` checks CUDA/NCCL gradient averaging, disjoint worker
CPU sets, and thread counts bounded by `SLURM_CPUS_PER_TASK`. It records the host task affinity
before entering Apptainer; Slurm abstract CPU IDs are never treated as Linux
CPU IDs. The report is `outputs/results/ou_preflight_JOBID.json`. It does not
exercise the simulator. `submit_ou.sh` accepts additional sbatch arguments.

An already-built full simulator SIF remains supported via `SAFETYMIMIC_SIF`.
The local Docker wrapper exposes GPU 0 only and is not used for two-GPU jobs.

## Assets and verification gates

Run these commands on an allocated HPC node. Use one `srun` task per command;
the numerical and simulator commands explicitly select the Slurm runtime.
Standalone verification entrypoints live in `tests/integration/` and run when
requested. Their filenames keep them outside default pytest discovery; simulator
and Slurm checks require their documented runtime. Environment setup also runs
`check_env.py`.

Manual gradient diagnostics live in `tests/debug/diagnose_layer.py`. They are
not collected by pytest. The archived replay utility is local-only and ignored;
maintained verification tools stay tracked so the HPC receives them through Git.

```bash
scripts/hpc/python.sh scripts/download_assets.py --prepare
scripts/hpc/python.sh tests/integration/validate_geometry.py
scripts/hpc/python.sh tests/integration/validate_starting_policy.py --runtime slurm
scripts/hpc/python.sh -m pytest -q
scripts/hpc/python.sh scripts/fit_world_model.py --runtime slurm
scripts/hpc/python.sh tests/integration/validate_world_model.py --runtime slurm
scripts/hpc/python.sh tests/integration/validate_layer.py --runtime slurm --finite_difference
```

Asset preparation downloads the pinned actor, motions and robot meshes, generates
MJCF/USD and ground assets, and writes the standing reference plus SHA-256 records.
The starting-policy check compares real pretrained actions with zero ball
conditioning and tests ball-free standing. World-model fitting uses the full
motion set and eight-step fitting windows; actor fine-tuning keeps its weights
fixed. Training and layer validation share their checkpoint and safety defaults.
A failed validation saves evidence and returns a failing exit code.

Before a long run, complete two optimizer updates in a fresh directory, resume
to a third update, test graceful stopping, and profile ten updates at the intended
horizon/batch. Check finite gradients, all actor groups changing, frozen model
weights, unchanged normalization, synchronized ranks and complete snapshots.
Inspect QP failure/relaxation rates separately from physical hits and falls.
Earlier test results do not validate a newly pulled revision.

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
