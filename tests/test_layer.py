import torch

from safety_mimic.ball import BallState, aimed_throw
from safety_mimic.layer import SafetyConfig, SafetyLayer
from safety_mimic.qp import QPResult


class NonlinearDynamics:
    def step(self, state, action, context):
        return state + action + 0.1 * action.square()


class LinearGeometry:
    def barriers(self, state, ball, margin):
        return state - ball.position[:, :1] - margin, torch.zeros_like(state)


class MovingGeometry:
    def barriers(self, state, ball, margin):
        return state - margin, torch.ones_like(state) * 3


def test_feasible_nominal_uses_exact_inactive_solution(monkeypatch):
    def unexpected_solve(*args, **kwargs):
        raise AssertionError("A feasible nominal requires no numerical QP solve")

    monkeypatch.setattr("safety_mimic.layer.project_action", unexpected_solve)
    layer = SafetyLayer(NonlinearDynamics(), LinearGeometry(), SafetyConfig(velocity_time=0))
    state = torch.tensor([[0.5]], dtype=torch.double)
    nominal = torch.tensor([[0.1]], dtype=torch.double, requires_grad=True)
    ball = BallState(
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, dtype=torch.double),
    )
    result = layer(state, ball, nominal, None, nominal.new_full((1, 1), -2), nominal.new_full((1, 1), 2))
    torch.testing.assert_close(result.qp.action, nominal, atol=0, rtol=0)
    gradient = torch.autograd.grad(result.qp.action.sum(), nominal)[0]
    torch.testing.assert_close(gradient, torch.ones_like(nominal), atol=0, rtol=0)
    assert not result.failed.any()


def test_velocity_augmentation_does_not_hide_geometric_violation():
    state = torch.tensor([[-0.1]], dtype=torch.double)
    ball = BallState(
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, dtype=torch.double),
    )
    layer = SafetyLayer(NonlinearDynamics(), MovingGeometry(), SafetyConfig(clearance_margin=0))
    barrier = layer.barrier(state, ball)
    assert barrier.shape == (1, 2)
    assert barrier[0, 0] < 0
    assert barrier[0, 1] > 0


def test_complete_layer_gradient_includes_linearization_dependence():
    layer = SafetyLayer(
        NonlinearDynamics(),
        LinearGeometry(),
        SafetyConfig(velocity_time=0, clearance_margin=0, slack_weight=100, trust_radius=2),
    )
    state = torch.tensor([[0.5]], dtype=torch.double, requires_grad=True)
    nominal = torch.tensor([[-0.7]], dtype=torch.double, requires_grad=True)
    ball = BallState(
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, dtype=torch.double),
    )

    def action(s, a):
        return layer(s, ball, a, None, a.new_full(a.shape, -2), a.new_full(a.shape, 2)).qp.action

    assert torch.autograd.gradcheck(action, (state, nominal), atol=3e-4, rtol=3e-3)


def test_backward_ablation_preserves_forward_action():
    layer = SafetyLayer(
        NonlinearDynamics(),
        LinearGeometry(),
        SafetyConfig(velocity_time=0, clearance_margin=0, slack_weight=100, trust_radius=2),
    )
    state = torch.tensor([[0.5]], dtype=torch.double)
    nominal = torch.tensor([[-0.7]], dtype=torch.double, requires_grad=True)
    ball = BallState(
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, dtype=torch.double),
    )
    bounds = (nominal.new_full(nominal.shape, -2), nominal.new_full(nominal.shape, 2))
    full = layer(state, ball, nominal, None, *bounds)
    surrogate = layer(state, ball, nominal, None, *bounds, backward="identity")
    torch.testing.assert_close(full.qp.action, surrogate.qp.action)
    gradient = torch.autograd.grad(surrogate.qp.action.sum(), nominal)[0]
    torch.testing.assert_close(gradient, torch.ones_like(nominal))


def test_execution_filter_has_no_actor_graph_or_higher_order_jacobians(monkeypatch):
    calls = []
    original = torch.autograd.grad

    def record(*args, **kwargs):
        calls.append(kwargs.get("create_graph", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", record)
    layer = SafetyLayer(NonlinearDynamics(), LinearGeometry(), SafetyConfig(
        velocity_time=0, clearance_margin=0, slack_weight=100, trust_radius=2))
    state = torch.tensor([[0.5]], dtype=torch.double, requires_grad=True)
    nominal = torch.tensor([[-0.7]], dtype=torch.double, requires_grad=True)
    ball = BallState(torch.zeros(1, 3, dtype=torch.double), torch.zeros(1, 3, dtype=torch.double),
                     torch.zeros(1, dtype=torch.double))
    result = layer.filter(state, ball, nominal, None, nominal.new_full((1, 1), -2), nominal.new_full((1, 1), 2))
    assert calls and not any(calls)
    for value in (result.qp.action, result.qp.slack, result.qp.affine_residual, result.nonlinear_residual,
                  result.clearance, result.initial_barrier):
        assert not value.requires_grad
    assert state.grad is nominal.grad is None
    assert not result.failed.any()


def test_seeded_ball_reaches_target():
    target = torch.tensor([[0.0, 0.0, 0.9], [0.0, 0.0, 0.5]])
    ball = aimed_throw(target, torch.Generator().manual_seed(17))
    repeated = aimed_throw(target, torch.Generator().manual_seed(17))
    torch.testing.assert_close(ball.position, repeated.position)
    torch.testing.assert_close(ball.step(0.7).position, target, atol=1e-6, rtol=1e-6)


def test_multiple_local_qps_keep_gradients_connected():
    layer = SafetyLayer(
        NonlinearDynamics(),
        LinearGeometry(),
        SafetyConfig(
            velocity_time=0,
            clearance_margin=0,
            slack_weight=100,
            trust_radius=2,
            linearization_steps=3,
            step_trust_radius=0.3,
            residual_margin=0.001,
        ),
    )
    state = torch.tensor([[0.5]], dtype=torch.double, requires_grad=True)
    nominal = torch.tensor([[-0.7]], dtype=torch.double, requires_grad=True)
    ball = BallState(
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, 3, dtype=torch.double),
        torch.zeros(1, dtype=torch.double),
    )

    def action(s, a):
        return layer(s, ball, a, None, a.new_full(a.shape, -2), a.new_full(a.shape, 2)).qp.action

    assert torch.autograd.gradcheck(action, (state, nominal), atol=3e-4, rtol=3e-3)


def test_mixed_batch_keeps_earlier_valid_candidate(monkeypatch):
    calls = []

    class Dynamics:
        def step(self, state, action, context):
            return action

    def solve(nominal, *unused, **kwargs):
        calls.append(None)
        action = nominal * 0 + nominal.new_tensor([[0.1], [-0.1]] if len(calls) == 1 else [[-0.3], [0.1]])
        return QPResult(action, action * 0, action, action[:, 0] < 0)

    monkeypatch.setattr("safety_mimic.layer.project_action", solve)
    nominal = torch.tensor([[-0.2], [-0.8]], dtype=torch.double, requires_grad=True)
    ball = BallState(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2))
    layer = SafetyLayer(Dynamics(), LinearGeometry(), SafetyConfig(
        velocity_time=0, clearance_margin=0, linearization_steps=2))
    result = layer(nominal.detach() * 0, ball, nominal, None, nominal * 0 - 2, nominal * 0 + 2)
    torch.testing.assert_close(result.qp.action, nominal.new_full((2, 1), 0.1))
    assert not result.failed.any()
    assert len(calls) == 2


def test_iteration_exhaustion_retains_nonlinear_failure():
    class ConcaveDynamics:
        def step(self, state, action, context):
            return 1 - action.square()

    nominal = torch.tensor([[2.0]], dtype=torch.double, requires_grad=True)
    ball = BallState(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1))
    state = nominal.detach() * 0
    bounds = nominal * 0 - 3, nominal * 0 + 3
    config = dict(velocity_time=0, clearance_margin=0, residual_margin=0.002, trust_radius=2)
    short = SafetyLayer(ConcaveDynamics(), LinearGeometry(), SafetyConfig(**config, linearization_steps=1))
    full = SafetyLayer(ConcaveDynamics(), LinearGeometry(), SafetyConfig(**config))
    assert short(state, ball, nominal, None, *bounds).failed.all()
    result = full(state, ball, nominal, None, *bounds)
    assert not result.failed.any()
    assert result.iterations.item() > 1
    gradient = torch.autograd.grad(result.qp.action.sum(), nominal)[0]
    assert torch.isfinite(gradient).all()


def test_stalled_search_preserves_failure_without_repeating_identical_qp(monkeypatch):
    calls = []

    def unchanged(nominal, *unused, **kwargs):
        calls.append(None)
        return QPResult(nominal, nominal * 0 + 0.1, nominal * 0 - 0.1,
                        torch.ones(len(nominal), dtype=torch.bool))

    monkeypatch.setattr("safety_mimic.layer.project_action", unchanged)
    layer = SafetyLayer(NonlinearDynamics(), LinearGeometry(), SafetyConfig(velocity_time=0))
    nominal = torch.tensor([[-0.7]], dtype=torch.double, requires_grad=True)
    ball = BallState(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1))
    result = layer(nominal * 0 + 0.5, ball, nominal, None, nominal * 0 - 2, nominal * 0 + 2)
    assert len(calls) == 1 and result.failed.all() and result.search_stalled.all()
