"""Fit ordinary SuperMimic dynamics from transitions of the released policy.

No policy optimization and no safety-aware dynamics loss are used here.
"""

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--window", type=int, default=4)
parser.add_argument("--hidden", nargs="+", type=int, default=[2048, 2048, 2048])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--action_noise", type=float, default=0.15)
parser.add_argument("--resume", default=None)
parser.add_argument("--motion_dir", default=str(ROOT / "data/standing"))
parser.add_argument("--output", default=str(ROOT / "outputs/checkpoints/world_model.pt"))
add_launcher_args(parser)
args = parser.parse_args()

import torch
from supermimic import networks
from supermimic.buffer import ReplayBuffer
from supermimic.checkpoint import load_checkpoint
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list

from safety_mimic.runtime import configure_runtime


def main():
    configure_runtime(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    networks.AMP_DTYPE = None
    device = "cuda:0"
    geometry = G1Kinematics(device)
    motion = MotionLib.from_dir(args.motion_dir, geometry, device)
    mdp = MDP(geometry, motion, torch.tensor(default_joint_pos_list(), device=device))
    checkpoint = load_checkpoint(ROOT / "checkpoints/g1-lafan1-040000", device)
    policy = Policy(OBS_DIM, 730).to(device).eval().requires_grad_(False)
    policy.load_state_dict(checkpoint["pi"])
    world_model = WorldModel(hidden=tuple(args.hidden)).to(device)
    if args.resume:
        previous = torch.load(args.resume, map_location=device, weights_only=True)
        world_model.load_state_dict(previous["wm"])
    optimizer = torch.optim.Adam(world_model.parameters(), lr=2e-4)
    buffer = ReplayBuffer(256, args.num_envs, device)
    floor = torch.cat(
        (
            torch.full((3,), 0.003),
            torch.full((3,), 0.005),
            torch.full((3,), 0.02),
            torch.full((3,), 0.05),
            torch.full((29,), 0.003),
            torch.full((29,), 0.05),
        )
    ).to(device)
    logs = []
    start = time.monotonic()
    sim_cfg = make_sim(device, args)
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, reset_noise=False)
        history = None
        for iteration in range(args.iterations + 64):
            with torch.no_grad():
                state = env.state()
                observation = mdp.obs(state, env.a_prev, env.xi, env.f)
                history = (
                    observation[:, None].repeat(1, 10, 1)
                    if history is None
                    else torch.cat((history[:, 1:], observation[:, None]), 1)
                )
                action = policy(history, motion.goal(env.f, 10)) + args.action_noise * torch.randn(
                    args.num_envs, 29, device=device
                )
                transition = env.step(action)
                buffer.add(transition)
                history[transition["done"]] = 0
                world_model.in_norm.update(world_model.inputs(state, action, normalize_xi(transition["xi"])))
                world_model.out_norm.update(world_model.targets(state, unflatten_state(transition["s1"])))
            if iteration < 64:
                continue
            sample = buffer.sample_windows(args.batch_size, args.window)
            predicted = unflatten_state(sample["s"][:, 0])
            error = 0
            for step in range(args.window):
                predicted = world_model.step(
                    predicted, sample["a"][:, step], normalize_xi(sample["xi"][:, step])
                )
                target = unflatten_state(sample["s1"][:, step])
                scaled = world_model.state_error(predicted, target) / world_model.out_norm.std().clamp_min(
                    floor
                )
                error = error + (scaled.square().mean(-1) * sample["mask"][:, step]).mean()
            loss = error / args.window
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
            optimizer.step()
            if iteration % 50 == 0:
                row = {
                    "iteration": iteration - 64,
                    "loss": float(loss.detach()),
                    "sim_transitions": (iteration + 1) * args.num_envs,
                    "wall_seconds": time.monotonic() - start,
                }
                logs.append(row)
                print(json.dumps(row), flush=True)
        torch.cuda.synchronize()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "wm": world_model.state_dict(),
                "hidden": args.hidden,
                "args": vars(args),
                "sim_transitions": (args.iterations + 64) * args.num_envs,
                "wall_seconds": time.monotonic() - start,
            },
            output,
        )
        output.with_suffix(".fit.json").write_text(json.dumps(logs, indent=2) + "\n")
        print(f"Saved ordinary dynamics model to {output}", flush=True)


if __name__ == "__main__":
    main()
