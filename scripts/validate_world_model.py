"""Check held-out clearance prediction and action sensitivities in physical simulation."""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from safety_mimic.settings import SAFETY_DEFAULTS, WORLD_MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runtime", choices=["local", "slurm"], default="local")
parser.add_argument("--cpu_threads", type=int)
parser.add_argument("--checkpoint", default=str(ROOT / WORLD_MODEL_PATH))
parser.add_argument("--pairs", type=int, default=16)
parser.add_argument("--samples", type=int, default=10)
parser.add_argument("--epsilon", type=float, default=0.05)
parser.add_argument("--velocity_time", type=float, default=SAFETY_DEFAULTS["velocity_time"])
parser.add_argument("--output", type=Path, default=ROOT / "outputs/results/world_model_validation.json")
add_launcher_args(parser)
args = parser.parse_args()

import torch
from supermimic import networks
from supermimic.checkpoint import load_checkpoint
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list

from safety_mimic.ball import BallState
from safety_mimic.geometry import CollisionGeometry
from safety_mimic.runtime import configure_runtime


def main():
    configure_runtime(args.device, profile=args.runtime, cpu_threads=args.cpu_threads)
    torch.manual_seed(10001)
    networks.AMP_DTYPE = None
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda:0"
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(str(ROOT / "data/standing"), kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    geometry = CollisionGeometry(
        ROOT / "vendor/supermimic/data/g1_description/g1_29dof.xml", G1_JOINT_NAMES, device=device
    )
    base = Policy(OBS_DIM, 730).to(device).eval().requires_grad_(False)
    base.load_state_dict(load_checkpoint(ROOT / "checkpoints/g1-lafan1-040000", device)["pi"])
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = WorldModel(hidden=checkpoint["hidden"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["wm"])
    records = []
    sim_cfg = make_sim(device, args)
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(sim_cfg, 2 * args.pairs, motion, mdp, eval_mode=True, reset_noise=False)
        ids = torch.arange(env.N, device=device)
        for sample in range(args.samples):
            env.reset(ids, torch.zeros(env.N, dtype=torch.long, device=device))
            env.refresh()
            history = None
            for step in range(15 + sample * 3):
                with torch.no_grad():
                    state = env.state()
                    observation = mdp.obs(state, env.a_prev, env.xi, env.f)
                    history = (
                        observation[:, None].repeat(1, 10, 1)
                        if history is None
                        else torch.cat((history[:, 1:], observation[:, None]), 1)
                    )
                    nominal = base(history, motion.goal(env.f, 10))
                    if step + 1 < 15 + sample * 3:
                        env.step(nominal)
            direction = torch.nn.functional.normalize(torch.randn(args.pairs, 29, device=device), dim=-1)
            signed_direction = torch.stack((direction, -direction), 1).flatten(0, 1)
            actions = nominal + args.epsilon * signed_direction
            with torch.no_grad():
                predicted = model.step(state, actions, normalize_xi(env.xi))
                centers, _ = geometry.centers_and_velocities(state)
                ball = BallState(
                    centers[:, 0] + state[0].new_tensor([1, 0, 0.2]),
                    state[0].new_tensor([-5, 0, -1]).expand(env.N, -1),
                    state[0].new_full((env.N,), 0.08),
                )
                predicted_h, predicted_rate = geometry.barriers(predicted, ball.step(0.02))
                env.step(actions)
                actual_h, actual_rate = geometry.barriers(env.state(), ball.step(0.02))
                predicted_features = predicted_h + args.velocity_time * predicted_rate
                actual_features = actual_h + args.velocity_time * actual_rate
                predicted_derivative = (predicted_features[0::2] - predicted_features[1::2]) / (
                    2 * args.epsilon
                )
                actual_derivative = (actual_features[0::2] - actual_features[1::2]) / (2 * args.epsilon)
                cosine = torch.nn.functional.cosine_similarity(
                    predicted_derivative, actual_derivative, dim=-1
                )
                relative = torch.linalg.vector_norm(predicted_derivative - actual_derivative, dim=-1) / (
                    torch.linalg.vector_norm(actual_derivative, dim=-1).clamp_min(1e-6)
                )
                records.append(
                    {
                        "sample": sample,
                        "clearance_rmse_m": float((predicted_h - actual_h).square().mean().sqrt()),
                        "derivative_cosine_mean": float(cosine.mean()),
                        "derivative_relative_error_mean": float(relative.mean()),
                        "pair_state_disagreement": float((state[4][0::2] - state[4][1::2]).abs().max()),
                    }
                )
        summary = {
            key: sum(record[key] for record in records) / len(records)
            for key in records[0]
            if key != "sample"
        }
        summary["passed"] = (
            summary["clearance_rmse_m"] < 0.02
            and summary["derivative_cosine_mean"] > 0.25
            and summary["derivative_relative_error_mean"] < 2
            and summary["pair_state_disagreement"] < 1e-5
        )
        summary["scope"] = "local pilot gate; not a certified model-error bound or global validation"
        result = {"summary": summary, "samples": records, "args": vars(args)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise RuntimeError("World-model validation failed; inspect the saved prediction and sensitivity errors")


if __name__ == "__main__":
    main()
