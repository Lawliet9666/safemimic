import copy
from types import SimpleNamespace

import pytest
import torch
from test_policy import TinyTracker
from torch import nn
from supermimic.networks import Critic

from safety_mimic.ball import BallState
from safety_mimic.layer import SafetyConfig, SafetyLayer
from safety_mimic.policy import BallConditionedPolicy
from safety_mimic.qp import QPResult
from safety_mimic.layer import SafetyResult
from safety_mimic.training import (
    LossConfig, clip_gradients, imagined_objective, lambda_return_targets, update_critic,
)


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))

    def step(self, state, action, context):
        p, r, v, w, q, qd = state
        shift = torch.cat((action[:, :1] + 0.1 * action[:, :1].square(), 0.05 * action[:, 1:3]), -1)
        return p + self.gain * shift, r, v, w, q + 0.01 * action, qd + 0.01 * action


class Geometry:
    def barriers(self, state, ball, margin=0.02):
        return state[0][:, :1] - margin, torch.zeros_like(state[0][:, :1])


class MDP:
    def obs(self, state, previous, xi, frame):
        return torch.cat(
            (state[4], state[5], state[0], state[2], previous, previous.new_ones(previous.shape[0], 6)), -1
        )

    def reward(self, state, frame, action, previous):
        return 3 - (state[0][:, 0] - 0.6).square() - 0.1 * (action - previous).square().mean(-1)

    def soft_termination(self, state, frame):
        return state[0].new_full((state[0].shape[0],), 0.05)

    def critic_features(self, state, frame, context):
        return torch.cat((state[0], state[4], state[5]), -1)


def setup():
    torch.manual_seed(12)
    actor = BallConditionedPolicy(tracker=TinyTracker()).double()
    with torch.no_grad():
        actor.tracker.dec.weight.mul_(0.03)
        actor.tracker.dec.bias.zero_()
        actor.tracker.dec.bias[0] = -0.7
    dynamics = Dynamics().double().requires_grad_(False)
    geometry = Geometry()
    layer = SafetyLayer(
        dynamics,
        geometry,
        SafetyConfig(velocity_time=0, clearance_margin=0, slack_weight=100, trust_radius=2),
    )
    state = (
        torch.tensor([[0.5, 0.0, 0.8]], dtype=torch.double),
        torch.eye(3, dtype=torch.double)[None],
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, 29, dtype=torch.double),
        torch.zeros(1, 29, dtype=torch.double),
    )
    sample = {
        "state": state,
        "history": torch.zeros(1, 10, 99, dtype=torch.double),
        "previous": torch.zeros(1, 29, dtype=torch.double),
        "frame": torch.zeros(1, dtype=torch.long),
        "xi": torch.zeros(1, 33, dtype=torch.double),
        "ball": BallState(
            torch.ones(1, 3, dtype=torch.double),
            torch.zeros(1, 3, dtype=torch.double),
            torch.zeros(1, dtype=torch.double),
        ),
    }
    motion = SimpleNamespace(
        q=torch.zeros(12, 29, dtype=torch.double),
        root_p=torch.tensor([[0, 0, 0.8]] * 12, dtype=torch.double),
        clip_end=torch.tensor([12]),
        goal=lambda frame, k: torch.ones(1, 7, dtype=torch.double),
    )
    bounds = torch.full((1, 29), -2.0, dtype=torch.double), torch.full((1, 29), 2.0, dtype=torch.double)
    return actor, dynamics, layer, MDP(), motion, geometry, sample, *bounds


def test_single_qp_rollout_trains_full_actor_while_world_model_is_fixed():
    arguments = setup()
    actor, dynamics = arguments[:2]
    frozen_before = dynamics.gain.detach().clone()
    visited_states = []
    handle = actor.register_forward_pre_hook(lambda module, inputs: visited_states.append(inputs[2][0].detach().clone()))
    loss, metrics, _ = imagined_objective(*arguments, LossConfig(horizon=2))
    handle.remove()
    assert metrics["correction"] > 0 and metrics["task"] < 0
    assert metrics["imagined_transitions"] == 2
    assert len(visited_states) == 2  # Exactly one actor evaluation per rollout step.
    assert visited_states[1][0, 0] > 0.3  # The next state follows the QP action, not the unsafe raw action.
    loss.backward()
    for module in actor.parameter_groups().values():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert dynamics.gain.grad is None
    torch.testing.assert_close(dynamics.gain, frozen_before)


@pytest.mark.parametrize("correction_weight,slack_weight", [(0, 0), (1, 10)])
def test_actual_qp_sensitivities_change_policy_gradient_without_changing_forward_loss(
    correction_weight, slack_weight,
):
    args = setup()
    other = (copy.deepcopy(args[0]), *args[1:])
    options = {
        "horizon": 1,
        "correction_weight": correction_weight,
        "slack_weight": slack_weight,
    }
    full, _, _ = imagined_objective(*args, LossConfig(**options))
    surrogate, _, _ = imagined_objective(*other, LossConfig(method="identity", **options))
    torch.testing.assert_close(full, surrogate, atol=1e-9, rtol=1e-9)
    full.backward()
    surrogate.backward()
    assert not torch.allclose(
        args[0].tracker.dec.bias.grad, other[0].tracker.dec.bias.grad, atol=1e-5, rtol=1e-5
    )


def test_reward_baseline_never_calls_qp_and_unfrozen_model_is_rejected():
    args = list(setup())
    args[2] = None
    loss, metrics, _ = imagined_objective(*args, LossConfig(method="reward", horizon=2))
    assert metrics["correction"] == 0 and metrics["qp_failure_rate"] == 0
    assert metrics["imagined_transitions"] == 2
    loss.backward()
    args[1].requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        imagined_objective(*args, LossConfig(method="reward"))


def test_external_filter_and_nominal_actor_updates_preserve_supermimic_gradient():
    args = list(setup())
    args[2] = None
    other = (copy.deepcopy(args[0]), *args[1:])
    baseline, baseline_metrics, _ = imagined_objective(*args, LossConfig(method="supermimic", horizon=2))
    filtered, filtered_metrics, _ = imagined_objective(*other, LossConfig(method="filter", horizon=2))
    torch.testing.assert_close(baseline, filtered, atol=0, rtol=0)
    baseline.backward()
    filtered.backward()
    for left, right in zip(args[0].parameters(), other[0].parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    for module in args[0].parameter_groups().values():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    for metrics in (baseline_metrics, filtered_metrics):
        assert metrics["correction"] == metrics["qp_slack"] == metrics["baseline_avoidance"] == 0
    assert all(p.grad is None for p in args[1].parameters())


def test_original_reward_alignment_discount_and_survival_are_preserved():
    args = list(setup())
    frames = []

    class ConstantMDP(MDP):
        def reward(self, state, frame, action, previous):
            frames.append(frame.clone())
            return state[0].new_full((len(frame),), 2)

        def soft_termination(self, state, frame):
            return state[0].new_full((len(frame),), 0.25)

    args[3] = ConstantMDP()
    loss, metrics, _ = imagined_objective(
        *args, LossConfig(horizon=2, gamma=0.9, correction_weight=0, slack_weight=0),
    )
    torch.testing.assert_close(loss, loss.new_tensor(-(2 + 0.9 * 0.75 * 2)))
    assert [int(frame[0]) for frame in frames] == [1, 2]
    assert metrics["final_survival"] == pytest.approx(0.75**2)


def test_terminal_critic_has_state_gradients_but_no_parameter_gradients():
    args = setup()
    other = (copy.deepcopy(args[0]), *args[1:])
    class LinearCritic(nn.Linear):
        def forward(self, features):
            return super().forward(features).squeeze(-1)

    critic = LinearCritic(74, 1, bias=False).double().requires_grad_(False)
    with torch.no_grad():
        critic.weight.zero_()
        critic.weight[0, 0] = 2
    config = LossConfig(horizon=1, correction_weight=0, slack_weight=0)
    with_value, metrics, batch = imagined_objective(*args, config, critic=critic)
    without_value, _, _ = imagined_objective(*other, config)
    assert metrics["terminal_value"] > 0
    torch.testing.assert_close(with_value + metrics["terminal_value"], without_value)
    with_value.backward()
    without_value.backward()
    assert critic.weight.grad is None
    assert not torch.allclose(args[0].tracker.dec.bias.grad, other[0].tracker.dec.bias.grad)
    assert batch.features.shape == (2, 1, 74)
    assert not batch.features.requires_grad and not batch.rewards.requires_grad
    # The last six critic features contain the relative ball state.
    assert batch.features[0, 0, -6:].abs().sum() > 0


def test_critic_update_changes_only_critic_and_uses_detached_lambda_returns():
    args = setup()
    critic = Critic(74).double().requires_grad_(False)
    _, _, batch = imagined_objective(*args, LossConfig(horizon=2), critic=critic)
    before = [p.clone() for p in critic.parameters()]
    critic.requires_grad_(True)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    value = update_critic(critic, optimizer, batch, epochs=2, minibatches=4)
    assert value >= 0 and value < float("inf")
    assert any(not torch.equal(a, b) for a, b in zip(before, critic.parameters()))
    assert all(p.grad is None for p in args[0].parameters())
    assert all(p.grad is None for p in args[1].parameters())


def test_lambda_returns_handle_bootstrap_lambda_and_termination():
    rewards = torch.tensor([[1.0], [2.0]])
    terminations = torch.zeros_like(rewards)
    values = torch.tensor([[7.0], [8.0], [10.0]])
    torch.testing.assert_close(
        lambda_return_targets(rewards, terminations, values, 0.9, 1), torch.tensor([[10.9], [11.0]]),
    )
    torch.testing.assert_close(
        lambda_return_targets(rewards, terminations, values, 0.9, 0), torch.tensor([[8.2], [11.0]]),
    )
    terminations[0] = 1
    torch.testing.assert_close(
        lambda_return_targets(rewards, terminations, values, 0.9, 1), torch.tensor([[1.0], [11.0]]),
    )


def test_slack_remains_differentiable_and_invalid_correction_teacher_is_excluded():
    args = list(setup())

    class RelaxedLayer:
        def __call__(self, state, ball, raw, *unused, **kwargs):
            slack = (raw[:, :1] + 2).square()
            qp = QPResult(raw * 0.5, slack, -slack, torch.ones(len(raw), dtype=torch.bool))
            return SafetyResult(qp, -100 * slack, slack, qp.failed)

    class ZeroRewardMDP(MDP):
        def reward(self, state, frame, action, previous):
            return action.sum(-1) * 0

    args[2], args[3] = RelaxedLayer(), ZeroRewardMDP()
    loss, metrics, _ = imagined_objective(*args, LossConfig(horizon=1, slack_weight=3))
    assert metrics["correction"] == 0 and metrics["correction_valid_rate"] == 0
    assert metrics["qp_slack"] > 0 and metrics["qp_violation"] > 0
    # Nonlinear violation is telemetry only, even when its value is enormous.
    assert float(loss.detach()) == pytest.approx(0.02 * 3 * metrics["qp_slack"])
    loss.backward()
    assert args[0].tracker.dec.bias.grad[0] != 0


def test_correction_detaches_teacher_but_not_the_actor():
    args = list(setup())
    raw_actions = []

    class ValidLayer:
        def __call__(self, state, ball, raw, *unused, **kwargs):
            raw.retain_grad()
            raw_actions.append(raw)
            zero = raw[:, :1] * 0
            qp = QPResult(raw * 0.5, zero, zero, torch.zeros(len(raw), dtype=torch.bool))
            return SafetyResult(qp, zero, zero, qp.failed)

    class ZeroRewardMDP(MDP):
        def reward(self, state, frame, action, previous):
            return action.sum(-1) * 0

    args[2], args[3] = ValidLayer(), ZeroRewardMDP()
    loss, metrics, _ = imagined_objective(*args, LossConfig(horizon=1, slack_weight=0))
    loss.backward()
    raw = raw_actions[0]
    torch.testing.assert_close(raw.grad, 0.02 * raw.detach() / raw.shape[-1])
    assert metrics["correction"] > 0


def test_large_finite_gradients_are_clipped_without_overflow_or_zeroing():
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3e30, 4e30])
    norm = clip_gradients([parameter], 1.0)
    assert torch.isfinite(norm)
    torch.testing.assert_close(norm, torch.tensor(5e30, dtype=torch.double))
    torch.testing.assert_close(parameter.grad, torch.tensor([0.6, 0.8]))


def test_nonfinite_global_gradient_is_rejected_before_optimizer_update():
    parameter = nn.Parameter(torch.zeros(1))
    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(RuntimeError, match="Nonfinite global"):
        clip_gradients([parameter])
