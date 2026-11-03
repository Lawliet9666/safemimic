"""Learned-model barrier construction with gradients through QP coefficients."""

from dataclasses import dataclass

import torch

from .ball import BallState
from .geometry import CollisionGeometry, RobotState
from .qp import QPResult, project_action


@dataclass
class SafetyConfig:
    dt: float = 0.02
    alpha: float = 0.2
    clearance_margin: float = 0.02
    residual_margin: float = 0.0
    velocity_time: float = 0.15
    geometric_constraint: bool = True
    anticipation_mode: str = "all"
    trust_radius: float = 1.0
    slack_weight: float = 1e4
    tolerance: float = 1e-4
    linearization_steps: int = 1
    step_trust_radius: float | None = None


@dataclass
class SafetyResult:
    qp: QPResult
    nonlinear_residual: torch.Tensor
    clearance: torch.Tensor
    failed: torch.Tensor


class SafetyLayer:
    """One-step local geometric and velocity-augmented barrier filter.

    The optional augmented barrier h + velocity_time * h_dot anticipates closing
    motion. The geometric inequality is retained independently: augmentation
    alone is not asserted to certify the geometric safe set. Physical-model and
    intersample error bounds are still needed for a physical safety guarantee.
    """

    def __init__(self, dynamics, geometry: CollisionGeometry, config: SafetyConfig | None = None):
        self.dynamics = dynamics
        self.geometry = geometry
        self.config = config or SafetyConfig()
        if (
            not 0 < self.config.alpha <= 1
            or self.config.dt <= 0
            or self.config.velocity_time < 0
            or self.config.linearization_steps < 1
            or self.config.anticipation_mode not in {"all", "threat"}
        ):
            raise ValueError("Invalid barrier parameters")

    def barrier(self, state: RobotState, ball: BallState):
        clearance, rate = self.geometry.barriers(state, ball, self.config.clearance_margin)
        if self.config.velocity_time == 0:
            return clearance
        if self.config.anticipation_mode == "threat":
            predicted = self.geometry.predictive_clearance(state, ball, self.config.clearance_margin)
            weight = torch.sigmoid((0.05 - predicted) / 0.03)
        else:
            weight = 1
        augmented = clearance + self.config.velocity_time * weight * rate
        return torch.cat((clearance, augmented), -1) if self.config.geometric_constraint else augmented

    def residual(self, state, ball, action, context):
        following = self.dynamics.step(state, action, context)
        return self.barrier(following, ball.step(self.config.dt)) - (1 - self.config.alpha) * self.barrier(
            state, ball
        )

    def __call__(self, state, ball, nominal, context, lower, upper, backward="implicit"):
        if backward not in {"implicit", "identity"}:
            raise ValueError("Backward must be implicit or identity")
        training_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            linearization = nominal if nominal.requires_grad else nominal.detach().requires_grad_(True)
            for step in range(self.config.linearization_steps):
                residual = self.residual(state, ball, linearization, context)
                if (
                    step == 0
                    and bool(torch.isfinite(residual).all())
                    and bool((residual >= self.config.residual_margin + self.config.tolerance).all())
                    and bool(((nominal >= lower) & (nominal <= upper)).all())
                ):
                    # With zero correction and zero slack, the feasible nominal
                    # is exactly the strongly convex QP's unique optimum. Its
                    # local action derivative is the identity. Avoid constructing
                    # unused model Jacobians and a dense QP in this open region.
                    qp = QPResult(
                        nominal,
                        torch.zeros_like(residual),
                        residual - self.config.residual_margin,
                        torch.zeros(nominal.shape[0], device=nominal.device, dtype=torch.bool),
                    )
                    nonlinear = residual
                    break
                # A batched VJP computes all constraint rows in one autograd
                # call. Separate environments must remain independent.
                basis = torch.eye(residual.shape[1], device=residual.device, dtype=residual.dtype)
                basis = basis[:, None, :].expand(-1, residual.shape[0], -1)
                jacobian = torch.autograd.grad(
                    residual,
                    linearization,
                    grad_outputs=basis,
                    is_grads_batched=True,
                    create_graph=training_graph,
                    retain_graph=True,
                )[0].permute(1, 0, 2)
                qp = project_action(
                    nominal,
                    jacobian,
                    residual,
                    lower,
                    upper,
                    linearization=linearization,
                    step_trust_radius=self.config.step_trust_radius,
                    trust_radius=self.config.trust_radius,
                    margin=self.config.residual_margin,
                    slack_weight=self.config.slack_weight,
                    tolerance=self.config.tolerance,
                )
                linearization = qp.action
                nonlinear = self.residual(state, ball, qp.action, context)
                if not bool((qp.failed | (nonlinear.amin(-1) < -self.config.tolerance)).any()):
                    break
            # The QP margin reserves room for approximation error. Validate the
            # original nonlinear condition separately from that affine reserve.
            clearance, _ = self.geometry.barriers(state, ball, self.config.clearance_margin)
            failed = (
                qp.failed | ~torch.isfinite(nonlinear).all(-1) | (nonlinear.amin(-1) < -self.config.tolerance)
            )
            if backward == "identity":
                qp.action = qp.action.detach() + (nominal - nominal.detach())
        if not training_graph:
            qp.action = qp.action.detach()
            qp.slack = qp.slack.detach()
            qp.affine_residual = qp.affine_residual.detach()
            nonlinear = nonlinear.detach()
            clearance = clearance.detach()
        return SafetyResult(qp, nonlinear, clearance, failed)
