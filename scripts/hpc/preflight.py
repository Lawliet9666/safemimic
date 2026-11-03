"""Verify an actual two-GPU Slurm allocation before starting long training."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from safety_mimic.distributed import Distribution
from safety_mimic.runtime import configure_runtime

device = configure_runtime(profile="slurm", cpu_threads=int(os.environ.get("CPU_THREADS", "16")))
distribution = Distribution(device)
try:
    if distribution.world_size != 2:
        raise ValueError("The OU preflight expects exactly two ranks")
    parameter = torch.nn.Parameter(torch.zeros(2, device=device))
    parameter.grad = torch.full_like(parameter, distribution.rank + 1)
    distribution.average_gradients([parameter])
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 1.5))
    records = [None] * distribution.world_size
    dist.all_gather_object(records, {
        "rank": distribution.rank, "device": device, "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "cpus": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
    })
    if distribution.primary:
        result = {"passed": True, "scope": "NCCL/gradient/CPU allocation; simulator not exercised",
                  "torch": str(torch.__version__), "cuda": torch.version.cuda, "ranks": records}
        output = Path("outputs/results") / f"ou_preflight_{os.environ['SLURM_JOB_ID']}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
finally:
    distribution.close()
