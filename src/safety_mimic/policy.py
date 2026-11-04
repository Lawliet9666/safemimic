"""Fully trainable SuperMimic actor with ball conditioning in its motion latent.

The zero-initialized ball encoder preserves the pretrained actor at startup.
The motion encoder, ball encoder, and shared action decoder are all optimized.
Nominal deployment needs neither dynamics weights nor a QP solver. The external
filter baseline requires both and records that contract in its checkpoint.
"""

from pathlib import Path

import torch
from supermimic.checkpoint import load_checkpoint
from supermimic.mdp import OBS_DIM
from supermimic.networks import Policy, run_mlp
from torch import nn

CHECKPOINT_FORMAT = "safetymimic.full_policy.v1"


def ball_features(state, ball):
    position, rotation, velocity = state[:3]
    transpose = rotation.transpose(-1, -2)
    relative_position = (transpose @ (ball.position - position).unsqueeze(-1)).squeeze(-1)
    relative_velocity = (transpose @ (ball.velocity - velocity).unsqueeze(-1)).squeeze(-1)
    return torch.cat((relative_position / 4, relative_velocity / 10), -1)


class BallConditionedPolicy(nn.Module):
    def __init__(self, *, ball_hidden=128, tracker=None):
        super().__init__()
        self.tracker = tracker if tracker is not None else Policy(OBS_DIM, 730)
        self.ball_hidden = ball_hidden
        self.ball_encoder = nn.Sequential(nn.Linear(6, ball_hidden), nn.ELU(), nn.Linear(ball_hidden, 32))
        nn.init.zeros_(self.ball_encoder[-1].weight)
        nn.init.zeros_(self.ball_encoder[-1].bias)

    @classmethod
    def from_pretrained(cls, directory, *, device="cpu", ball_hidden=128):
        actor = cls(ball_hidden=ball_hidden).to(device)
        actor.tracker.load_state_dict(load_checkpoint(Path(directory), device)["pi"], strict=True)
        return actor.requires_grad_(True)

    def forward(self, history, goal, state, ball):
        motion_latent = run_mlp(self.tracker.enc, self.tracker.goal_norm(goal))
        ball_latent = self.ball_encoder(ball_features(state, ball))
        latent = torch.tanh(motion_latent + ball_latent)
        observation = self.tracker.obs_norm(history).flatten(-2)
        return run_mlp(self.tracker.dec, torch.cat((latent, observation), -1))

    def parameter_groups(self):
        return {
            "motion_encoder": self.tracker.enc,
            "action_decoder": self.tracker.dec,
            "ball_encoder": self.ball_encoder,
        }


def save_policy(path, policy, *, config, training):
    """Save self-contained actor weights; no pretrained files needed to reload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "format": CHECKPOINT_FORMAT,
        "architecture": {"ball_hidden": policy.ball_hidden},
        "policy": policy.state_dict(),
        "config": config,
        "training": training,
        "deployment": {"uses_qp": config.get("method") == "filter",
                       "uses_world_model": config.get("method") == "filter"},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, temporary)
    temporary.replace(path)


def load_policy(path, *, device="cpu"):
    record = torch.load(path, map_location=device, weights_only=True)
    if record.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Expected {CHECKPOINT_FORMAT}; old residual checkpoints are not supported")
    policy = BallConditionedPolicy(**record["architecture"]).to(device)
    policy.load_state_dict(record["policy"], strict=True)
    return policy, {key: value for key, value in record.items() if key != "policy"}


def update_history(history, observation, reset=None):
    if history is None:
        return observation[:, None].repeat(1, 10, 1)
    following = torch.cat((history[:, 1:], observation[:, None]), 1)
    if reset is not None:
        following = torch.where(reset[:, None, None], observation[:, None].expand_as(following), following)
    return following
