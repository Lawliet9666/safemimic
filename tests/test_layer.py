import torch

from safety_mimic.ball import BallState, aimed_throw
from safety_mimic.layer import SafetyConfig, SafetyLayer


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
