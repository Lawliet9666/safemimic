"""Check runtime imports and allow only the documented upstream metadata overrides."""

import argparse
import importlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "torch": "2.11.0+cu128", "numpy": "2.3.5", "numba": "0.64.0", "mujoco": "3.15.0",
    "warp-lang": "1.17.0+cu12", "packaging": "26.0", "newton-usd-schemas": "0.5.0",
}
# IsaacLab's pinned root pyproject explicitly overrides its importers' old
# MuJoCo/USD/packaging pins. The repository Dockerfile installs qpth --no-deps;
# its NumPy cap is checked separately from the tested Torch gradient path.
OVERRIDES = {
    ("mujoco-usd-converter", "mujoco"): "3.15.0",
    ("qpth", "numpy"): "2.3.5",
    ("isaacsim-asset-isolated", "usd-exchange"): "3.0.1",
    ("isaacsim-asset-isolated", "mujoco"): "3.15.0",
    ("isaacsim-asset-isolated", "usd-optimize"): "1.3.1",
    ("ovphysx", "packaging"): "26.0",
    ("isaacsim-robot-schema", "usd-exchange"): "3.0.1",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/results/environment.json")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux":
        raise RuntimeError("The pinned simulator requires Linux and Python 3.12")
    if Version(platform.libc_ver()[1]) < Version("2.35"):
        raise RuntimeError("Use scripts/hpc/python.sh: the simulator needs glibc >=2.35")
    versions = {name: metadata.version(name) for name in EXPECTED}
    errors = [f"{name}: expected {version}, got {versions[name]}"
              for name, version in EXPECTED.items() if versions[name] != version]
    accepted = []
    for distribution in metadata.distributions():
        name = canonicalize_name(distribution.metadata["Name"])
        for declaration in distribution.requires or []:
            requirement = Requirement(declaration)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                errors.append(f"{name}: missing {requirement.name}")
                continue
            if requirement.specifier and installed not in requirement.specifier:
                message = f"{name}: {requirement}, installed {installed}"
                if OVERRIDES.get((name, canonicalize_name(requirement.name))) == installed:
                    accepted.append(message)
                else:
                    errors.append(message)
    pins = json.loads((ROOT / "source_pins.json").read_text())
    for name, path in (("supermimic", ROOT / "vendor/supermimic"),
                       ("isaaclab", ROOT / "vendor/supermimic/IsaacLab")):
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        if revision != pins[name]["revision"]:
            errors.append(f"{name}: source revision differs from source_pins.json")
    for module in ("numba", "torch", "mujoco", "warp", "qpth", "isaaclab.app",
                   "isaaclab_newton.physics", "supermimic.env", "safety_mimic.training"):
        importlib.import_module(module)
    result = {"passed": not errors, "versions": versions, "documented_metadata_overrides": accepted,
              "errors": errors, "scope": "imports and version checks; asset conversion and simulation are separate gates"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if errors:
        raise RuntimeError("Environment validation failed; inspect the saved report")


if __name__ == "__main__":
    main()
