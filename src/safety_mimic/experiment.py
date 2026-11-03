"""Policy-only deployment. Never imports dynamics or safety layers."""

from pathlib import Path

import torch
from supermimic import networks
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list

from .geometry import CollisionGeometry
from .policy import BallConditionedPolicy, load_policy, update_history


def resolve_artifact(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def action_bounds(geometry, env):
    lower = (geometry.q_lo - env.default_q - env.xi[:, 1:30]) / env.alpha
    upper = (geometry.q_hi - env.default_q - env.xi[:, 1:30]) / env.alpha
    return lower.clamp_min(-5), upper.clamp_max(5)


class Controller:
    def __init__(
        self, root, *, checkpoint=None, pretrained="checkpoints/g1-lafan1-040000", seed=0, device="cuda:0"
    ):
        self.root, self.device = Path(root), device
        torch.manual_seed(seed)
        networks.AMP_DTYPE = None
        self.checkpoint = resolve_artifact(root, checkpoint) if checkpoint else None
        if self.checkpoint:
            self.policy, saved = load_policy(self.checkpoint, device=device)
            self.method = saved["config"]["method"]
        else:
            self.policy = BallConditionedPolicy.from_pretrained(
                resolve_artifact(root, pretrained), device=device
            )
            self.method, saved = "pretrained", {}
        self.policy.eval().requires_grad_(False)
        self.kin = G1Kinematics(device)
        self.motion = MotionLib.from_dir(str(self.root / "data/standing"), self.kin, device)
        self.mdp = MDP(self.kin, self.motion, torch.tensor(default_joint_pos_list(), device=device))
        self.geometry = CollisionGeometry(
            self.root / "vendor/supermimic/data/g1_description/g1_29dof.xml", G1_JOINT_NAMES, device=device
        )
        self.metadata = {
            "method": self.method,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "training": saved.get("training", {}),
            "training_config": saved.get("config", {}),
            "deployment": {"uses_qp": False, "uses_world_model": False},
        }

    def bounds(self, env):
        return action_bounds(self.geometry, env)

    def nominal(self, env, history):
        state, ball = env.state(), env.ball_state()
        observation = self.mdp.obs(state, env.a_prev, env.xi, env.f)
        history = update_history(history, observation, env.ep_step == 0)
        return self.policy(history, self.motion.goal(env.f, 10), state, ball), history

    def execute(self, env, nominal):
        lower, upper = self.bounds(env)
        return nominal.clamp(lower, upper)
