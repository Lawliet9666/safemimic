"""Record the actual source and checkpoint bytes used by an experiment."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def write_provenance(directory: Path, root: Path, checkpoints=()):
    files = sorted((root / "src/safety_mimic").glob("*.py")) + sorted((root / "scripts").glob("*.py"))
    files += [root / "scripts/container.sh", root / "Dockerfile"]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    for checkpoint in checkpoints:
        path = Path(checkpoint)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashes,
        "source_pins": json.loads((root / "source_pins.json").read_text()),
        "asset_pins": json.loads((root / "asset_pins.json").read_text()),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "allowed_cpus": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0),
        "visible_cuda_devices": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (directory / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
