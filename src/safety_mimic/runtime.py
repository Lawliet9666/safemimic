"""The project's GPU 0 / six logical CPU resource contract."""

import hashlib
import os

import torch


def configure_runtime(device="cuda:0"):
    if device != "cuda:0":
        raise ValueError("SafetyMimic is restricted to cuda:0")
    cpus = set(os.sched_getaffinity(0)) & set(range(6))
    if not cpus:
        raise RuntimeError("CPUs 0-5 must be available")
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(min(6, len(cpus)))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True


def state_digest(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
