import torch

from safety_mimic.qp import project_action


def solve(nominal, jacobian, residual):
    return project_action(
        nominal,
        jacobian,
        residual,
        nominal.new_full(nominal.shape, -2),
        nominal.new_full(nominal.shape, 2),
        trust_radius=3,
        slack_weight=100,
        tolerance=1e-5,
    )


def test_joint_constraints_and_inactive_projection():
    nominal = torch.tensor([[-0.5, 0.4], [0.7, 0.8]], dtype=torch.double)
    jacobian = torch.eye(2, dtype=torch.double).expand(2, -1, -1)
    residual = nominal - torch.tensor([0.2, -0.3], dtype=torch.double)
    result = solve(nominal, jacobian, residual)
    torch.testing.assert_close(
        result.action, torch.tensor([[0.2, 0.4], [0.7, 0.8]], dtype=torch.double), atol=1e-5, rtol=1e-5
    )
    assert not result.failed.any()


def test_impossible_constraint_is_explicit_failure():
    nominal = torch.zeros(1, 1, dtype=torch.double)
    result = solve(nominal, torch.zeros(1, 1, 1, dtype=torch.double), -torch.ones_like(nominal))
    assert result.failed.item()
    assert result.slack.item() >= 0.999
    assert torch.isfinite(result.action).all()


def test_redundant_barriers_with_more_constraints_than_actions():
    nominal = torch.tensor([[-0.4]], dtype=torch.double, requires_grad=True)
    jacobian = torch.ones(1, 4, 1, dtype=torch.double)
    residual = (nominal - 0.2).expand(1, 4)
    result = solve(nominal, jacobian, residual)
    torch.testing.assert_close(result.action, nominal.new_full((1, 1), 0.2), atol=1e-5, rtol=1e-5)
    gradient = torch.autograd.grad(result.action.sum(), nominal)[0]
    assert torch.isfinite(gradient).all()
    assert not result.failed.any()


def test_implicit_derivatives_include_constraint_coefficients():
    nominal = torch.tensor([[-0.5, 0.4]], dtype=torch.double, requires_grad=True)
    jacobian = torch.tensor([[[1.2, 0.3]]], dtype=torch.double, requires_grad=True)
    residual = torch.tensor([[-0.7]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda n, j, r: solve(n, j, r).action, (nominal, jacobian, residual), atol=2e-4, rtol=2e-3
    )
