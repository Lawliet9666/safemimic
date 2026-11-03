"""Reference batched QP with implicit differentiation and explicit failure masks."""

from dataclasses import dataclass

import torch
from qpth.qp import QPFunction


@dataclass
class QPResult:
    action: torch.Tensor
    slack: torch.Tensor
    affine_residual: torch.Tensor
    failed: torch.Tensor


def project_action(
    nominal: torch.Tensor,
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    linearization: torch.Tensor | None = None,
    step_trust_radius: float | None = None,
    trust_radius: float = 1.0,
    margin: float = 0.0,
    slack_weight: float = 1e4,
    tolerance: float = 1e-5,
) -> QPResult:
    """Project onto J(a-linearization)+residual >= margin (default anchor: nominal).

    A nonnegative slack per barrier makes infeasible states diagnosable. Any
    slack larger than tolerance is a safety failure, even if the relaxed QP
    converges. No gradient or solver exception silently returns the nominal.
    Bounds are hard; the local trust box expands to include its clipped center.
    """
    if trust_radius <= 0 or slack_weight <= 0:
        raise ValueError("Trust radius and slack weight must be positive")
    batch, actions = nominal.shape
    constraints = residual.shape[1]
    point = nominal if linearization is None else linearization
    lo = torch.maximum(lower.expand_as(nominal), nominal.clamp(lower, upper) - trust_radius)
    hi = torch.minimum(upper.expand_as(nominal), nominal.clamp(lower, upper) + trust_radius)
    if step_trust_radius is not None:
        if step_trust_radius <= 0:
            raise ValueError("Local step trust radius must be positive")
        center = point.clamp(lower, upper)
        lo = torch.maximum(lo, center - step_trust_radius)
        hi = torch.minimum(hi, center + step_trust_radius)
    if bool((lo >= hi).any()):
        raise ValueError("Action bounds must have a nonempty interior")
    # Solve in double precision; cast keeps the entire graph connected.
    nominal_d, jacobian_d, residual_d = nominal.double(), jacobian.double(), residual.double()
    eye_a = torch.eye(actions, device=nominal.device, dtype=torch.double).expand(batch, -1, -1)
    eye_c = torch.eye(constraints, device=nominal.device, dtype=torch.double).expand(batch, -1, -1)
    zeros_ac = nominal_d.new_zeros(batch, actions, constraints)
    zeros_ca = zeros_ac.transpose(1, 2)
    # Linear exact penalty admits zero slack at a feasible optimum. A purely
    # quadratic slack penalty would intentionally violate every active barrier.
    q = torch.eye(actions + constraints, device=nominal.device, dtype=torch.double)
    q = q.expand(batch, -1, -1).contiguous()
    linear = torch.cat((-nominal_d, nominal_d.new_full((batch, constraints), slack_weight)), -1)
    matrix = torch.cat(
        (
            torch.cat((-jacobian_d, -eye_c), -1),
            torch.cat((eye_a, zeros_ac), -1),
            torch.cat((-eye_a, zeros_ac), -1),
            torch.cat((zeros_ca, -eye_c), -1),
        ),
        1,
    )
    rhs = torch.cat(
        (
            residual_d - (jacobian_d @ point.double().unsqueeze(-1)).squeeze(-1) - margin,
            hi.double(),
            -lo.double(),
            nominal_d.new_zeros(batch, constraints),
        ),
        -1,
    )
    empty = nominal_d.new_empty(0)
    solution = QPFunction(verbose=-1, eps=1e-10, maxIter=60)(q, linear, matrix, rhs, empty, empty)
    action = solution[:, :actions].to(nominal.dtype)
    slack = solution[:, actions:].to(nominal.dtype)
    affine = residual + (jacobian @ (action - point).unsqueeze(-1)).squeeze(-1) - margin
    failed = (
        ~torch.isfinite(solution).all(-1)
        | (slack.amax(-1) > tolerance)
        | (affine.amin(-1) < -tolerance)
        | ((action < lo - tolerance) | (action > hi + tolerance)).any(-1)
    )
    return QPResult(action, slack, affine, failed)
