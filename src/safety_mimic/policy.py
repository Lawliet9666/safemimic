"""Fully trainable SuperMimic actor with ball conditioning in its motion latent.

The zero-initialized ball encoder preserves the pretrained actor at startup.
Training selects whether to optimize the actor or only the optional class-K head.
Nominal deployment needs neither dynamics weights nor a QP solver. The external
filter baseline requires both and records that contract in its checkpoint.
"""

from pathlib import Path

import torch
from supermimic.checkpoint import load_checkpoint
from supermimic.mdp import OBS_DIM
from supermimic.networks import Policy, run_mlp
from torch import nn

from .gain import ClassKGainHead

CHECKPOINT_FORMAT = "safetymimic.full_policy.v1"


def ball_features(state, ball):
    position, rotation, velocity = state[:3]
    transpose = rotation.transpose(-1, -2)
    relative_position = (transpose @ (ball.position - position).unsqueeze(-1)).squeeze(-1)
    relative_velocity = (transpose @ (ball.velocity - velocity).unsqueeze(-1)).squeeze(-1)
    return torch.cat((relative_position / 4, relative_velocity / 10), -1)


class BallConditionedPolicy(nn.Module):
    def __init__(self, *, ball_hidden=128, tracker=None, adaptive_gain=False,
                 gain_hidden=32, gain_min=0.05, gain_max=0.8, gain_initial=0.2):
        super().__init__()
        self.tracker = tracker if tracker is not None else Policy(OBS_DIM, 730)
        self.ball_hidden = ball_hidden
        self.ball_encoder = nn.Sequential(nn.Linear(6, ball_hidden), nn.ELU(), nn.Linear(ball_hidden, 32))
        nn.init.zeros_(self.ball_encoder[-1].weight)
        nn.init.zeros_(self.ball_encoder[-1].bias)
        self.gain_head = None
        if adaptive_gain:
            self.enable_gain_head(gain_hidden, gain_min, gain_max, gain_initial)

    def enable_gain_head(self, hidden=32, minimum=0.05, maximum=0.8, initial=0.2):
        if self.gain_head is not None:
            raise ValueError("The policy already has a class-K gain head")
        reference = next(self.tracker.parameters())
        self.gain_head = ClassKGainHead(OBS_DIM + 32 + 6, hidden, minimum, maximum, initial).to(reference)

    @classmethod
    def from_pretrained(cls, directory, *, device="cpu", ball_hidden=128, **kwargs):
        actor = cls(ball_hidden=ball_hidden, **kwargs).to(device)
        actor.tracker.load_state_dict(load_checkpoint(Path(directory), device)["pi"], strict=True)
        return actor.requires_grad_(True)

    def _features(self, history, goal, state, ball):
        motion_latent = run_mlp(self.tracker.enc, self.tracker.goal_norm(goal))
        ball_latent = self.ball_encoder(ball_features(state, ball))
        latent = torch.tanh(motion_latent + ball_latent)
        observation = self.tracker.obs_norm(history)
        return latent, observation

    def gain_features(self, history, goal, state, ball):
        latent, observation = self._features(history, goal, state, ball)
        return torch.cat((latent, observation[:, -1], ball_features(state, ball)), -1)

    def forward_with_gain(self, history, goal, state, ball):
        latent, observation = self._features(history, goal, state, ball)
        action = run_mlp(self.tracker.dec, torch.cat((latent, observation.flatten(-2)), -1))
        gain = None if self.gain_head is None else self.gain_head(
            torch.cat((latent, observation[:, -1], ball_features(state, ball)), -1))
        return action, gain

    def forward(self, history, goal, state, ball, *, return_gain=False):
        outputs = self.forward_with_gain(history, goal, state, ball)
        return outputs if return_gain else outputs[0]

    def parameter_groups(self):
        groups = {
            "motion_encoder": self.tracker.enc,
            "action_decoder": self.tracker.dec,
            "ball_encoder": self.ball_encoder,
        }
        if self.gain_head is not None:
            groups["gain_head"] = self.gain_head
        return groups


def save_policy(path, policy, *, config, training):
    """Save self-contained actor weights; no pretrained files needed to reload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    architecture = {"ball_hidden": policy.ball_hidden}
    if policy.gain_head is not None:
        head = policy.gain_head
        architecture.update(adaptive_gain=True, gain_hidden=head.hidden, gain_min=head.minimum,
                            gain_max=head.maximum, gain_initial=head.initial)
    uses_qp = policy.gain_head is not None or config.get("deployment_uses_qp", config.get("method") == "filter")
    record = {
        "format": CHECKPOINT_FORMAT,
        "architecture": architecture,
        "policy": policy.state_dict(),
        "config": config,
        "training": training,
        "deployment": {"uses_qp": uses_qp, "uses_world_model": uses_qp},
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
