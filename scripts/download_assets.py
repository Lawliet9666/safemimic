"""Download public source assets and checkpoint with immutable revision metadata."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    existing = json.loads((ROOT / "asset_pins.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_revision", default=existing["policy"]["revision"])
    parser.add_argument("--dataset_revision", default=existing["dataset"]["revision"])
    args = parser.parse_args()
    api = HfApi()
    policy_id = "pantelismanouselis/g1-supermimic-lafan1"
    dataset_id = "lvhaidong/LAFAN1_Retargeting_Dataset"
    policy_sha = api.model_info(policy_id, revision=args.policy_revision).sha
    dataset_sha = api.dataset_info(dataset_id, revision=args.dataset_revision).sha
    output = ROOT / "checkpoints" / "g1-lafan1-040000"
    snapshot_download(
        policy_id,
        revision=policy_sha,
        local_dir=output,
        allow_patterns=["policy.safetensors", "config.json", "manifest.json"],
    )
    manifest = json.loads((output / "manifest.json").read_text())
    for name in ("policy.safetensors", "config.json"):
        digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
        if digest != manifest[name]:
            raise ValueError(f"SHA-256 mismatch for {name}")
    source = ROOT / "data" / "source"
    snapshot_download(
        dataset_id,
        repo_type="dataset",
        revision=dataset_sha,
        local_dir=source,
        allow_patterns=["g1/*.csv", "robot_description/g1/*", "robot_description/g1/meshes/*"],
    )
    upstream_data = ROOT / "vendor" / "supermimic" / "data"
    upstream_data.mkdir(parents=True, exist_ok=True)
    for name, origin in (
        ("lafan1_g1", source / "g1"),
        ("g1_description", source / "robot_description" / "g1"),
    ):
        destination = upstream_data / name
        if not destination.exists():
            destination.symlink_to(os.path.relpath(origin, destination.parent), target_is_directory=True)
    pins = {
        "policy": {"repo": policy_id, "revision": policy_sha, "sha256": manifest},
        "dataset": {"repo": dataset_id, "revision": dataset_sha},
        "world_model_in_policy_release": False,
    }
    (ROOT / "asset_pins.json").write_text(json.dumps(pins, indent=2) + "\n")
    print(json.dumps(pins, indent=2))


if __name__ == "__main__":
    main()
