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
    linearization_steps: int = 16
    step_trust_radius: float | None = 0.15
    backtracking_steps: int = 8


@dataclass
class SafetyResult:
    qp: QPResult
    nonlinear_residual: torch.Tensor
    clearance: torch.Tensor
    failed: torch.Tensor
    iterations: torch.Tensor | None = None
    initial_residual: torch.Tensor | None = None
    initial_barrier: torch.Tensor | None = None
    search_stalled: torch.Tensor | None = None


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
            or self.config.backtracking_steps < 0
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

    def filter(self, state, ball, nominal, context, lower, upper):
        """Execute a black-box filter, retaining only local action Jacobians.

        No actor/QP training graph escapes this call. Failed finite relaxed
        solutions retain their failure mask; they are never labeled safe.
        """
        with torch.no_grad():
            result = self(state, ball, nominal, context, lower, upper)
        if not bool(torch.isfinite(result.qp.action).all()):
            raise RuntimeError("Nonfinite execution-filter action; simulator was not advanced")
        return result

    def __call__(self, state, ball, nominal, context, lower, upper, backward="implicit"):
        if backward not in {"implicit", "identity"}:
            raise ValueError("Backward must be implicit or identity")
        training_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            # These quantities are invariant across local QPs and backtracking.
            # Cache their graphs as well as values; never detach state gradients.
            current_barrier = self.barrier(state, ball)
            next_ball = ball.step(self.config.dt)

            def evaluate(action):
                following = self.dynamics.step(state, action, context)
                return self.barrier(following, next_ball) - (1 - self.config.alpha) * current_barrier

            linearization = nominal if nominal.requires_grad else nominal.detach().requires_grad_(True)
            best_qp = best_nonlinear = best_score = initial_residual = None
            best_valid = torch.zeros(nominal.shape[0], device=nominal.device, dtype=torch.bool)
            stalled = torch.zeros_like(best_valid)
            iterations = torch.zeros_like(best_valid, dtype=torch.long)
            for step in range(self.config.linearization_steps):
                residual = evaluate(linearization)
                if initial_residual is None:
                    initial_residual = residual.detach()
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
                    best_qp, best_nonlinear = qp, nonlinear
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
                nonlinear = evaluate(qp.action)
                valid = ~qp.failed & torch.isfinite(nonlinear).all(-1) & (
                    nonlinear.amin(-1) >= -self.config.tolerance
                )
                score = (
                    torch.relu(self.config.residual_margin - nonlinear).sum(-1)
                    + qp.slack.clamp_min(0).sum(-1)
                    + torch.relu(-qp.affine_residual).sum(-1)
                ).detach()
                score = torch.nan_to_num(score, nan=float('inf'), posinf=float('inf'))
                take = torch.ones_like(valid) if best_score is None else (
                    (valid & ~best_valid) | ((valid == best_valid) & (score < best_score))
                )
                if best_qp is None:
                    best_qp, best_nonlinear, best_score = qp, nonlinear, score
                else:
                    best_qp = QPResult(
                        torch.where(take[:, None], qp.action, best_qp.action),
                        torch.where(take[:, None], qp.slack, best_qp.slack),
                        torch.where(take[:, None], qp.affine_residual, best_qp.affine_residual),
                        torch.where(take, qp.failed, best_qp.failed),
                    )
                    best_nonlinear = torch.where(take[:, None], nonlinear, best_nonlinear)
                    best_score = torch.where(take, score, best_score)
                best_valid = best_valid | valid
                iterations = torch.where(~best_valid | take, iterations.new_full((), step + 1), iterations)
                if bool(best_valid.all()):
                    break
                # Globalize the next linearization with the true nonlinear
                # merit. Only the discrete step choice is detached. The selected
                # iterate still depends on the QP solution and all coefficients.
                with torch.no_grad():
                    current_merit = torch.relu(self.config.residual_margin - residual).sum(-1)
                    current_merit += 0.5 / self.config.slack_weight * (linearization - nominal).square().sum(-1)
                    fraction = nominal.new_zeros(nominal.shape[0])
                    found = best_valid.clone()
                    for backtrack in range(self.config.backtracking_steps + 1):
                        scale = 0.5**backtrack
                        point = linearization + scale * (qp.action - linearization)
                        value = evaluate(point)
                        merit = torch.relu(self.config.residual_margin - value).sum(-1)
                        merit += 0.5 / self.config.slack_weight * (point - nominal).square().sum(-1)
                        accept = ~found & torch.isfinite(merit) & (merit < current_merit - 1e-9)
                        fraction = torch.where(accept, fraction.new_full((), scale), fraction)
                        found |= accept
                        if bool(found.all()):
                            break
                stalled = ~best_valid & (fraction == 0)
                if bool((best_valid | stalled).all()):
                    # Re-solving at an unchanged point supplies the identical
                    # local problem. Keep its failure and avoid wasted solves.
                    break
                linearization = linearization + fraction[:, None] * (qp.action - linearization)
            qp, nonlinear = best_qp, best_nonlinear
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
            current_barrier = current_barrier.detach()
        return SafetyResult(qp, nonlinear, clearance, failed, iterations, initial_residual, current_barrier, stalled & failed)
