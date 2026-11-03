"""Fetch immutable upstream revisions and apply the small scene-factory hook."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments, cwd=None):
    subprocess.run(arguments, cwd=cwd, check=True)


def main():
    pins = json.loads((ROOT / "source_pins.json").read_text())
    for key, destination in (
        ("supermimic", ROOT / "vendor/supermimic"),
        ("isaaclab", ROOT / "vendor/supermimic/IsaacLab"),
    ):
        pin = pins[key]
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            run("git", "clone", "--depth", "1", pin["repository"], str(destination))
        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()
        if current != pin["revision"]:
            run("git", "fetch", "--depth", "1", "origin", pin["revision"], cwd=destination)
            run("git", "checkout", "--detach", pin["revision"], cwd=destination)
    source = ROOT / "vendor/supermimic"
    patch = ROOT / "patches/scene_factory.patch"
    applied = (
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)], cwd=source, capture_output=True, check=False
        ).returncode
        == 0
    )
    if not applied:
        run("git", "apply", "--check", str(patch), cwd=source)
        run("git", "apply", str(patch), cwd=source)


if __name__ == "__main__":
    main()
