"""Create fixed, shared development and held-out validation throws."""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("outputs/manifests"))
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np
from supermimic.env import make_sim

from safety_mimic.evaluation import REGIONS, dump_json, make_manifest, validate_manifest
from safety_mimic.experiment import Controller
from safety_mimic.isaac_env import DodgeballSim
from safety_mimic.runtime import configure_runtime

ROOT = Path(__file__).resolve().parents[1]


def main():
    configure_runtime(args.device)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a fresh manifest directory; frozen throws are never overwritten")
    controller = Controller(ROOT)
    sim_cfg = make_sim("cuda:0", args)
    sim_cfg.dt, sim_cfg.physics.num_substeps = 0.005, 1
    with launch_simulation(sim_cfg, args):
        env = DodgeballSim(sim_cfg, 8, controller.motion, controller.mdp, controller.geometry, seed=10000)
        initial = {
            "state": [v[0].cpu().tolist() for v in env.state()],
            "xi": env.xi[0].cpu().tolist(),
            "frame": int(env.f[0]),
            "previous_action": env.a_prev[0].cpu().tolist(),
        }
        geometry = controller.geometry
        centers, _ = geometry.centers_and_velocities(env.state())
        links = [geometry.model.body(int(geometry.model.geom_bodyid[g])).name for g in geometry.geom_ids]
        targets = {region: centers[0, links.index(region)].cpu().tolist() for region in REGIONS}
        args.output.mkdir(parents=True, exist_ok=True)
        for split in ("development", "validation"):
            manifest = make_manifest(split, initial, targets)
            validate_manifest(manifest)
            dump_json(args.output / (split + ".json"), manifest)
        # Store the analytic ballistic references explicitly, including their sampling times.
        for split in ("development", "validation"):
            import json

            m = json.loads((args.output / (split + ".json")).read_text())
            time = np.arange(151) * 0.02
            p = np.array([r["ball_position"] for r in m["throws"]])
            v = np.array([r["ball_velocity"] for r in m["throws"]])
            trajectories = p[:, None] + v[:, None] * time[None, :, None]
            trajectories += 0.5 * np.array([0, 0, -9.81]) * time[None, :, None] ** 2
            np.savez_compressed(
                args.output / (split + "_ballistic.npz"),
                time=time,
                position=trajectories,
                throw_id=[r["throw_id"] for r in m["throws"]],
            )
        print("Created development=96, validation=192 throws", flush=True)


if __name__ == "__main__":
    main()
