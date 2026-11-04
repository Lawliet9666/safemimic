import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from safety_mimic.ball import BallState
from safety_mimic.policy import BallConditionedPolicy, load_policy, save_policy, update_history


class TinyTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(7, 32)
        self.dec = nn.Linear(10 * 99 + 32, 29)
        self.goal_norm = nn.Identity()
        self.obs_norm = nn.Identity()

    def forward(self, history, goal):
        return self.dec(torch.cat((torch.tanh(self.enc(goal)), history.flatten(-2)), -1))


def inputs():
    state = (
        torch.randn(2, 3),
        torch.eye(3).repeat(2, 1, 1),
        torch.randn(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 29),
        torch.zeros(2, 29),
    )
    ball = BallState(torch.randn(2, 3), torch.randn(2, 3), torch.full((2,), 0.08))
    return torch.randn(2, 10, 99), torch.randn(2, 7), state, ball


def test_ball_branch_preserves_pretrained_network_at_initialization():
    torch.manual_seed(4)
    actor = BallConditionedPolicy(tracker=TinyTracker())
    history, goal, state, ball = inputs()
    torch.testing.assert_close(
        actor(history, goal, state, ball), actor.tracker(history, goal), atol=0, rtol=0
    )
    assert all(p.requires_grad for p in actor.parameters())


def test_optimizer_updates_original_encoder_decoder_and_ball_branch():
    torch.manual_seed(5)
    actor = BallConditionedPolicy(tracker=TinyTracker())
    before = copy.deepcopy(actor.state_dict())
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    data = inputs()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        actor(*data).square().mean().backward()
        for module in actor.parameter_groups().values():
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        optimizer.step()
    for prefix in ("tracker.enc", "tracker.dec", "ball_encoder.0", "ball_encoder.2"):
        assert any(
            not torch.equal(value, before[key])
            for key, value in actor.state_dict().items()
            if key.startswith(prefix)
        )


@pytest.mark.parametrize("method,uses_filter", [("safetymimic", False), ("filter", True)])
def test_checkpoint_round_trip_is_self_contained_and_rejects_old_residual(tmp_path, monkeypatch, method, uses_filter):
    torch.manual_seed(7)
    actor = BallConditionedPolicy()
    with torch.no_grad():
        actor.ball_encoder[-1].weight.normal_(std=0.02)
    path = tmp_path / "policy.pt"
    save_policy(path, actor, config={"method": method}, training={"updates": 2})

    def forbidden(*args, **kwargs):
        raise AssertionError("A complete checkpoint must not reload the pretrained tracker")

    monkeypatch.setattr("safety_mimic.policy.load_checkpoint", forbidden)
    restored, metadata = load_policy(path)
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    assert metadata["deployment"] == {"uses_qp": uses_filter, "uses_world_model": uses_filter}
    torch.save({"residual": {}}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="old residual"):
        load_policy(tmp_path / "old.pt")


def test_deployment_import_does_not_load_qp():
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root / "vendor/supermimic"))),
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import safety_mimic.experiment; "
                "assert 'qpth.qp' not in sys.modules; assert 'safety_mimic.layer' not in sys.modules"
            ),
        ],
        env=environment,
        check=True,
    )


def test_partial_reset_history_repeats_current_observation():
    history = torch.randn(2, 10, 99)
    observation = torch.randn(2, 99)
    following = update_history(history, observation, torch.tensor([True, False]))
    torch.testing.assert_close(following[0], observation[0].expand(10, 99))
    torch.testing.assert_close(following[1, :-1], history[1, 1:])
