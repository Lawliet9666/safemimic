import copy

import pytest
import torch
from test_layer import LinearGeometry, NonlinearDynamics
from test_policy import TinyTracker, inputs
from test_training import setup

from safety_mimic.ball import BallState
from safety_mimic.gain import ClassKGainHead, discounted_returns, gain_policy_loss
from safety_mimic.layer import SafetyConfig, SafetyLayer
from safety_mimic.policy import BallConditionedPolicy, load_policy, save_policy
from safety_mimic.training import LossConfig, imagined_objective


def test_gain_head_preserves_nominal_action_and_initial_fixed_coefficient():
    torch.manual_seed(8)
    actor = BallConditionedPolicy(tracker=TinyTracker())
    data = inputs()
    original = actor(*data).detach()
    actor.enable_gain_head()
    action, gain = actor(*data, return_gain=True)
    torch.testing.assert_close(action, original, atol=0, rtol=0)
    torch.testing.assert_close(gain, gain.new_full((2,), 0.2), atol=2e-8, rtol=0)
    extreme = actor.gain_head.transform(torch.tensor([-1e6, 1e6]))
    assert extreme[0] > 0 and extreme[1] <= 1


@pytest.mark.parametrize("minimum,maximum,initial", [(0, 1, .2), (.1, 1.1, .2), (.2, .8, .2)])
def test_invalid_gain_bounds_fail_explicitly(minimum, maximum, initial):
    with pytest.raises(ValueError):
        ClassKGainHead(3, minimum=minimum, maximum=maximum, initial=initial)


@pytest.mark.parametrize("method", ["filter", "safetymimic"])
def test_adaptive_checkpoint_restores_gain_and_execution_contract(tmp_path, method):
    actor = BallConditionedPolicy(adaptive_gain=True)
    with torch.no_grad():
        actor.gain_head.net[-1].weight.normal_(std=.01)
    path = tmp_path / "policy.pt"
    save_policy(path, actor, config={"method": method}, training={})
    restored, metadata = load_policy(path)
    assert restored.gain_head is not None
    assert metadata["deployment"] == {"uses_qp": True, "uses_world_model": True}
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)


def test_gain_gradient_matches_finite_differences_and_filter_has_no_graph():
    layer = SafetyLayer(NonlinearDynamics(), LinearGeometry(), SafetyConfig(
        velocity_time=0, clearance_margin=0, slack_weight=100, trust_radius=2))
    state = torch.tensor([[.5]], dtype=torch.double)
    nominal = torch.tensor([[-.7]], dtype=torch.double)
    gain = torch.tensor([.2], dtype=torch.double, requires_grad=True)
    ball = BallState(torch.zeros(1, 3, dtype=torch.double), torch.zeros(1, 3, dtype=torch.double),
                     torch.zeros(1, dtype=torch.double))
    bounds = nominal.new_full((1, 1), -2), nominal.new_full((1, 1), 2)

    def action(coefficient):
        return layer(state, ball, nominal, None, *bounds, gain=coefficient).qp.action

    assert torch.autograd.gradcheck(action, (gain,), atol=3e-4, rtol=3e-3)
    assert torch.autograd.grad(action(gain).sum(), gain)[0].abs().sum() > 0
    torch.testing.assert_close(action(gain), layer(state, ball, nominal, None, *bounds).qp.action)
    filtered = layer.filter(state, ball, nominal, None, *bounds, gain=gain)
    torch.testing.assert_close(filtered.qp.action, action(gain))
    assert not filtered.qp.action.requires_grad and not filtered.qp.slack.requires_grad
    for bad in (gain * 0, gain * 0 + 2, gain[:, None]):
        with pytest.raises(ValueError):
            action(bad)


def test_only_gain_head_can_learn_through_differentiable_qp_and_frozen_world_model():
    args = list(setup())
    actor = args[0]
    actor.enable_gain_head()
    actor.requires_grad_(False)
    actor.gain_head.requires_grad_(True)
    before = copy.deepcopy(actor.state_dict())
    loss, _, _ = imagined_objective(*args, LossConfig(horizon=1))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.gain_head.parameters())
    assert all(p.grad is None for name, module in actor.parameter_groups().items() if name != "gain_head"
               for p in module.parameters())
    assert all(p.grad is None for p in args[1].parameters())
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_score_function_learns_from_detached_outcomes_without_a_qp_backward():
    torch.manual_seed(42)
    frozen = torch.nn.Linear(2, 2).double().requires_grad_(False)
    features = frozen(torch.randn(128, 2, dtype=torch.double)).detach()
    head = ClassKGainHead(2).double()
    with torch.no_grad():
        distribution = head.distribution(features, .4)
        samples = distribution.sample()
        old = distribution.log_prob(samples)
        # A black-box response whose return increases with the executed gain.
        executed = head.transform(samples).detach()
        advantages = (executed - executed.mean()) / executed.std()
        initial = head(features).mean()
    optimizer = torch.optim.Adam(head.parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = gain_policy_loss(head, features, samples, old, advantages)
        loss.backward()
        assert head.net[-1].bias.grad.abs().sum() > 0
        optimizer.step()
    assert head(features).mean() > initial
    assert all(p.grad is None for p in frozen.parameters())
    with pytest.raises(ValueError, match="detached"):
        gain_policy_loss(head, features.requires_grad_(), samples, old, advantages)


def test_full_episode_returns_do_not_omit_fall_or_terminal_rewards():
    rewards = torch.tensor([[1., 1.], [-20., 1.], [0., 5.]])
    expected = torch.tensor([[-9., 2.75], [-20., 3.5], [0., 5.]])
    torch.testing.assert_close(discounted_returns(rewards, .5), expected)
