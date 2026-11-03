"""Select a quiet upright source pose and hold it for a station-holding pilot."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "checkpoints/g1-lafan1-040000/config.json").read_text())
    default = []
    for name in config["control"]["joint_names"]:
        value = 0.0
        for key, number in (
            ("hip_pitch", -0.312),
            ("knee", 0.669),
            ("ankle_pitch", -0.363),
            ("elbow", 0.6),
            ("shoulder_pitch", 0.2),
        ):
            if key in name:
                value = number
        if "shoulder_roll" in name:
            value = 0.2 if name.startswith("left") else -0.2
        default.append(value)
    best = None
    for path in sorted((ROOT / "data/source/g1").glob("*.csv")):
        data = np.loadtxt(path, delimiter=",")
        velocity = np.gradient(data[:, :3], 1 / 30, axis=0)
        joint_velocity = np.gradient(data[:, 7:], 1 / 30, axis=0)
        # CSV quaternions are xyzw. Penalize roll/pitch and departure from a
        # known bent-knee standing posture, plus source-frame motion.
        score = (
            np.square(data[:, 7:] - default).mean(-1)
            + 0.01 * np.square(joint_velocity).mean(-1)
            + np.square(velocity).sum(-1)
            + 5 * np.square(data[:, 3:5]).sum(-1)
        )
        score[(data[:, 2] < 0.68) | (data[:, 2] > 0.85)] = np.inf
        frame = int(score.argmin())
        if np.isfinite(score[frame]) and (best is None or score[frame] < best[0]):
            best = (float(score[frame]), path.name, frame, data[frame].copy())
    if best is None:
        raise ValueError("No upright standing candidate in source motions")
    score, name, frame, pose = best
    pose[:2] = 0
    output = ROOT / "data/standing"
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "stand.csv", np.repeat(pose[None], 360, axis=0), delimiter=",")
    metadata = {
        "source_clip": name,
        "source_frame_30hz": frame,
        "selection_score": score,
        "description": "held source pose; stable tracking must be checked experimentally",
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/standing_reference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
