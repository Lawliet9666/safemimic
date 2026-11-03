"""Give each worker independent compilation caches before importing simulation."""

import os
import runpy
from pathlib import Path

root = Path(__file__).resolve().parents[2]
cache = root / ".cache" / f"job_{os.environ['SLURM_JOB_ID']}" / f"rank_{os.environ['LOCAL_RANK']}"
os.environ["WARP_CACHE_PATH"] = str(cache / "warp")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
runpy.run_path(str(root / "scripts/train.py"), run_name="__main__")
