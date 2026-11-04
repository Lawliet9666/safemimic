"""Verify pretrained action preservation and ball-free standing in the real simulator."""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runtime", choices=["local", "slurm"], default="local")
parser.add_argument("--cpu_threads", type=int)
parser.add_argument("--pretrained", type=Path, default=ROOT / "checkpoints/g1-lafan1-040000")
parser.add_argument("--checkpoint", type=Path, help="Evaluate a trained actor without its execution filter")
parser.add_argument("--report_only", action="store_true", help="Report failed standing outcomes for a comparison gate")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--output", type=Path, default=ROOT / "outputs/results/pretrained_standing.json")
add_launcher_args(parser)
args = parser.parse_args()

import torch
from supermimic import networks
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, MAX_STEP_REWARD
from supermimic.motion import MotionLib
from supermimic.robot import default_joint_pos_list

from safety_mimic.ball import BallState
from safety_mimic.policy import BallConditionedPolicy, load_policy, update_history
from safety_mimic.runtime import configure_runtime, state_digest


def main():
    device = configure_runtime(args.device, profile=args.runtime, cpu_threads=args.cpu_threads)
    if args.steps < 1 or args.num_envs < 1:
        raise ValueError("Steps and simulator batch must be positive")
    torch.manual_seed(0)
    networks.AMP_DTYPE = None
    checkpoint = args.checkpoint if args.checkpoint is None or args.checkpoint.is_absolute() else ROOT / args.checkpoint
    policy = (load_policy(checkpoint, device=device)[0] if checkpoint else
              BallConditionedPolicy.from_pretrained(args.pretrained, device=device)).eval().requires_grad_(False)
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(str(ROOT / "data/standing"), kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    sim_cfg = make_sim(device, args)
    history = None
    failed = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    returns = torch.zeros(args.num_envs, device=device, dtype=torch.float64)
    minimum_height = torch.full((args.num_envs,), float("inf"), device=device)
    minimum_tilt_cos = torch.ones(args.num_envs, device=device)
    maximum_difference = 0.0
    with launch_simulation(sim_cfg, args), torch.no_grad():
        env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, eval_mode=True, reset_noise=False)
        env.reset(torch.arange(args.num_envs, device=device), torch.zeros(args.num_envs, device=device, dtype=torch.long))
        env.refresh()
        for _ in range(args.steps):
            state = env.state()
            history = update_history(history, mdp.obs(state, env.a_prev, env.xi, env.f))
            goal = motion.goal(env.f, 10)
            ball = BallState(state[0] + state[0].new_tensor([4, 0, 1]), torch.zeros_like(state[0]),
                             state[0].new_full((args.num_envs,), 0.08))
            action = policy(history, goal, state, ball)
            if checkpoint is None:
                original = policy.tracker(history, goal)
                torch.testing.assert_close(action, original, atol=0, rtol=0)
                maximum_difference = max(maximum_difference, float((action - original).abs().max()))
            if not torch.isfinite(action).all():
                raise RuntimeError("Nonfinite pretrained action")
            transition = env.step(action)
            failed |= transition["term"]
            returns += transition["r"]
            current = env.state()
            minimum_height = torch.minimum(minimum_height, current[0][:, 2])
            minimum_tilt_cos = torch.minimum(minimum_tilt_cos, current[1][:, 2, 2])
        normalized = returns / (args.steps * MAX_STEP_REWARD)
        result = {"passed": not bool(failed.any()), "steps": args.steps, "num_envs": args.num_envs,
                  "checkpoint": str(checkpoint) if checkpoint else None, "policy_sha256": state_digest(policy),
                  "action_preservation_checked": checkpoint is None,
                  "action_max_difference": maximum_difference if checkpoint is None else None,
                  "tracking_failures": failed.tolist(), "normalized_tracking_returns": normalized.tolist(),
                  "mean_normalized_tracking_return": float(normalized.mean()),
                  "minimum_root_height": minimum_height.tolist(), "minimum_tilt_cos": minimum_tilt_cos.tolist(),
                  "final_root_height": env.state()[0][:, 2].tolist(),
                  "scope": "ball-free standing; no QP, world model or policy optimization"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["passed"] and not args.report_only:
        raise RuntimeError("Pretrained standing failed; inspect the reference and simulator before fine-tuning")


if __name__ == "__main__":
    main()
