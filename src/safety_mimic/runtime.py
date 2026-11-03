"""Local GPU-0/six-core limits, or explicitly selected Slurm allocations."""

import hashlib
import os

import torch


def configure_runtime(device="cuda:0", *, profile="local", cpu_threads=None):
    allowed = sorted(os.sched_getaffinity(0))
    if profile == "local":
        if device != "cuda:0" or int(os.environ.get("WORLD_SIZE", 1)) != 1:
            raise ValueError("Local runs are restricted to one process on cuda:0")
        cpus = set(allowed) & set(range(6))
        threads = min(cpu_threads or 6, 6, len(cpus))
    elif profile == "slurm":
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("The Slurm runtime requires a SLURM_JOB_ID allocation")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        if not 0 <= local_rank < local_size or len(allowed) < local_size:
            raise ValueError("Invalid rank/CPU allocation")
        # A single srun task launches torchrun workers. Partition its assigned
        # CPU set; never assume Slurm assigned physical CPUs starting at zero.
        width = len(allowed) // local_size
        cpus = set(allowed[local_rank * width:(local_rank + 1) * width])
        threads = min(cpu_threads or width, len(cpus))
        device = f"cuda:{local_rank}"
    else:
        raise ValueError("Runtime profile must be local or slurm")
    if not cpus or threads < 1:
        raise RuntimeError("No usable CPUs in the requested allocation")
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(device)
    return device


def state_digest(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
