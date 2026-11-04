"""SuperMimic actor-critic return plus QP correction and slack, on one rollout."""

import math
from dataclasses import dataclass

import torch
from supermimic.mdp import normalize_xi
from supermimic.motion import CONTROL_DT

from .ball import BallState
from .policy import ball_features

OBJECTIVE_FORMAT = "safetymimic.supermimic_correction_slack.v1"


def objective_format(method):
    # Old filter snapshots optimized tracking alone and cannot resume this loss.
    return "safetymimic.filter_ball_avoidance.v1" if method == "filter" else OBJECTIVE_FORMAT


@dataclass
class LossConfig:
    method: str = "safetymimic"
    horizon: int = 32
    gamma: float = 0.99
    correction_weight: float = 1.0
    slack_weight: float = 10.0
    baseline_avoidance_weight: float = 10.0

    def __post_init__(self):
        if (self.method not in {"safetymimic", "identity", "reward", "supermimic", "filter"}
                or self.horizon < 1 or not 0 <= self.gamma <= 1):
            raise ValueError("Invalid training method or horizon")
        if any(
            not math.isfinite(getattr(self, key)) or getattr(self, key) < 0
            for key in ("correction_weight", "slack_weight", "baseline_avoidance_weight")
        ):
            raise ValueError("Loss weights must be finite and nonnegative")


@dataclass
class CriticBatch:
    features: torch.Tensor  # (H+1, B, features); detached from the actor graph.
    rewards: torch.Tensor  # (H, B), including the two auxiliary costs.
    terminations: torch.Tensor  # (H, B), differentiable termination probabilities, detached here.


def critic_input(mdp, motion, state, frame, xi, ball):
    return torch.cat((
        mdp.critic_features(state, frame, normalize_xi(xi)),
        motion.goal(frame, 10),
        ball_features(state, ball),
    ), -1)


def lambda_return_targets(rewards, terminations, values, gamma, gae_lambda):
    """Original SuperMimic bootstrapped lambda returns, not Monte Carlo labels."""
    target = values[-1]
    targets = []
    for t in range(len(rewards) - 1, -1, -1):
        target = rewards[t] + gamma * (1 - terminations[t]) * (
            (1 - gae_lambda) * values[t + 1] + gae_lambda * target
        )
        targets.append(target)
    return torch.stack(targets[::-1])


def actor_optimizer(policy, actor_lr, ball_lr):
    if any(not math.isfinite(value) or value <= 0 for value in (actor_lr, ball_lr)):
        raise ValueError("Actor and ball learning rates must be finite and positive")
    groups = policy.parameter_groups()
    return torch.optim.Adam([
        {"params": groups["motion_encoder"].parameters(), "lr": actor_lr, "name": "motion_encoder"},
        {"params": groups["action_decoder"].parameters(), "lr": actor_lr, "name": "action_decoder"},
        {"params": groups["ball_encoder"].parameters(), "lr": ball_lr, "name": "ball_encoder"},
    ], betas=(0.9, 0.95))


def combine_critic_batches(batches):
    return CriticBatch(*(torch.cat([getattr(batch, key) for batch in batches], 1)
                         for key in ("features", "rewards", "terminations")))


def update_critic(
    critic, optimizer, batch, *, gamma=0.99, gae_lambda=0.95, epochs=5, minibatches=4,
    distribution=None,
):
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1 or epochs < 1 or minibatches < 1:
        raise ValueError("Invalid critic update settings")
    if any(value.requires_grad for value in (batch.features, batch.rewards, batch.terminations)):
        raise ValueError("Critic training data must be detached from the actor rollout")
    with torch.no_grad():
        features_for_norm = batch.features.flatten(0, 1)
        if distribution is not None:
            features_for_norm = distribution.gather_tensor(features_for_norm)
        critic.norm.update(features_for_norm)
        values = critic(batch.features)
        targets = lambda_return_targets(batch.rewards, batch.terminations, values, gamma, gae_lambda)
    features = batch.features[:-1].flatten(0, 1)
    targets = targets.flatten()
    losses = []
    for _ in range(epochs):
        permutation = torch.randperm(len(targets), device=features.device)
        for indices in permutation.tensor_split(min(minibatches, len(targets))):
            error = None
            try:
                loss = (critic(features[indices]) - targets[indices]).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite critic loss; update was not applied")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
            except Exception as caught:
                error = caught
            if distribution is not None:
                distribution.agree(error)
                distribution.average_gradients(critic.parameters())
            elif error is not None:
                raise error
            clip_gradients(critic.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    optimizer.zero_grad(set_to_none=True)
    return sum(losses) / len(losses)


def imagined_objective(
    policy, world_model, safety, mdp, motion, geometry, sample, lower, upper, config, *, critic=None,
):
    if any(parameter.requires_grad for parameter in world_model.parameters()):
        raise ValueError("World-model parameters must be frozen during policy training")
    if critic is not None and any(parameter.requires_grad for parameter in critic.parameters()):
        raise ValueError("Critic parameters must be frozen during the actor update")
    imagined_filter = config.method in {"safetymimic", "identity"}
    if imagined_filter and safety is None:
        raise ValueError("CBF-guided training requires a differentiable safety layer")
    predicted, ball = sample["state"], sample["ball"]
    history, previous = sample["history"], sample["previous"]
    frame, xi = sample["frame"], sample["xi"]
    context = normalize_xi(xi)
    zero = predicted[0].new_zeros(())
    metrics = {key: zero for key in (
        "task", "correction", "qp_slack", "qp_violation", "action_saturation", "qp_failure_rate",
        "correction_valid_rate", "baseline_avoidance", "nonzero_valid_correction_rate",
        "qp_affine_failure_rate", "qp_nonlinear_failure_rate", "qp_relaxation_rate", "qp_iterations",
        "qp_initial_safe_rate", "qp_initial_geometric_safe_rate",
        "qp_search_stalled_rate",
    )}
    loss = zero
    alive = predicted[0].new_ones(lower.shape[0])
    features, rewards, terminations = [], [], []
    if critic is not None:
        features.append(critic_input(mdp, motion, predicted, frame, xi, ball).detach())
    for step in range(config.horizon):
        next_ball = ball.step(CONTROL_DT)
        next_frame = torch.minimum(frame + 1, motion.clip_end[0] - 1)
        raw = policy(history, motion.goal(frame, 10), predicted, ball)
        metrics["action_saturation"] = metrics["action_saturation"] + (
            (raw < lower) | (raw > upper)
        ).float().mean()
        correction = slack = raw.new_zeros(raw.shape[0])
        if not imagined_filter:
            executed = raw.clamp(lower, upper)
        else:
            backward = "identity" if config.method == "identity" else "implicit"
            result = safety(predicted, ball, raw, context, lower, upper, backward=backward)
            executed = result.qp.action
            # A failed/relaxed action is not a validated safety teacher. Its
            # finite trajectory and differentiable slack still inform training.
            valid = ~result.failed
            correction = (raw - executed.detach()).square().mean(-1) * valid.to(raw.dtype)
            slack = result.qp.slack.clamp_min(0).square().mean(-1)
            metrics["correction_valid_rate"] = metrics["correction_valid_rate"] + valid.float().mean()
            metrics["qp_failure_rate"] = metrics["qp_failure_rate"] + result.failed.float().mean()
            metrics["nonzero_valid_correction_rate"] = metrics["nonzero_valid_correction_rate"] + (
                valid & ((executed - raw).square().sum(-1) > 1e-10)
            ).float().mean()
            tolerance = getattr(getattr(safety, "config", None), "tolerance", 1e-4)
            for key, flag in (
                ("qp_affine_failure_rate", result.qp.affine_residual.amin(-1) < -tolerance),
                ("qp_nonlinear_failure_rate", result.nonlinear_residual.amin(-1) < -tolerance),
                ("qp_relaxation_rate", result.qp.slack.amax(-1) > tolerance),
            ):
                metrics[key] = metrics[key] + flag.float().mean()
            if result.iterations is not None:
                metrics["qp_iterations"] = metrics["qp_iterations"] + result.iterations.float().mean()
            if result.initial_barrier is not None:
                metrics["qp_initial_safe_rate"] = metrics["qp_initial_safe_rate"] + (
                    result.initial_barrier.amin(-1) >= 0
                ).float().mean()
                metrics["qp_initial_geometric_safe_rate"] = metrics["qp_initial_geometric_safe_rate"] + (
                    result.clearance.amin(-1) >= 0
                ).float().mean()
            if result.search_stalled is not None:
                metrics["qp_search_stalled_rate"] = metrics["qp_search_stalled_rate"] + result.search_stalled.float().mean()
            metrics["qp_violation"] = metrics["qp_violation"] + torch.relu(
                -result.nonlinear_residual
            ).square().mean().detach()
        if not bool(torch.isfinite(executed).all() & torch.isfinite(slack).all()):
            raise RuntimeError("Nonfinite QP action/slack; imagined model was not advanced")
        following = world_model.step(predicted, executed, context)
        # Reuse all eight original reward terms, aligned to the next reference
        # frame, before the method-specific auxiliary costs below.
        task_reward = mdp.reward(following, next_frame, executed, previous)
        if config.method in {"reward", "filter"}:
            # The external filter does not participate in this actor graph.
            # Learn avoidance directly through frozen dynamics and geometry.
            clearance, _ = geometry.barriers(following, next_ball)
            avoidance = torch.relu(0.05 - clearance).square().amax(-1)
            task_reward = task_reward - CONTROL_DT * config.baseline_avoidance_weight * avoidance
            metrics["baseline_avoidance"] = metrics["baseline_avoidance"] + avoidance.mean().detach()
        reward = task_reward - CONTROL_DT * (
            config.correction_weight * correction + config.slack_weight * slack
        )
        termination = mdp.soft_termination(following, next_frame)
        loss = loss - (config.gamma**step * alive * reward).mean()
        alive = alive * (1 - termination)
        for key, value in (("task", -task_reward), ("correction", correction), ("qp_slack", slack)):
            metrics[key] = metrics[key] + value.mean().detach()
        rewards.append(reward.detach())
        terminations.append(termination.detach())
        observation = mdp.obs(following, executed, xi, next_frame)
        history = torch.cat((history[:, 1:], observation[:, None]), 1)
        predicted, previous, ball, frame = following, executed, next_ball, next_frame
        if critic is not None:
            features.append(critic_input(mdp, motion, predicted, frame, xi, ball).detach())
    terminal = zero
    batch = None
    if critic is not None:
        # Freeze critic weights, not its derivative with respect to the final
        # imagined state. Critic fitting below uses detached copies only.
        value = critic(critic_input(mdp, motion, predicted, frame, xi, ball))
        if value.shape != alive.shape:
            raise ValueError("Critic must return one value per environment")
        terminal = (config.gamma**config.horizon * alive * value).mean()
        loss = loss - terminal
        batch = CriticBatch(torch.stack(features), torch.stack(rewards), torch.stack(terminations))
    metrics = {key: float(value / config.horizon) for key, value in metrics.items()}
    metrics["loss"] = float(loss.detach())
    metrics["terminal_value"] = float(terminal.detach())
    metrics["final_survival"] = float(alive.detach().mean())
    metrics["imagined_transitions"] = config.horizon * lower.shape[0]
    return loss, metrics, batch


def sample_replay(replay, batch_size):
    sample = replay[int(torch.randint(len(replay), ()))]
    indices = torch.randperm(sample["frame"].shape[0], device=sample["frame"].device)[:batch_size]
    return {
        "state": tuple(value[indices] for value in sample["state"]),
        "ball": BallState(
            *(getattr(sample["ball"], key)[indices] for key in ("position", "velocity", "radius"))
        ),
        **{key: sample[key][indices] for key in ("history", "frame", "xi", "previous")},
    }, indices


def parameter_gradient_norm(parameters):
    """Accumulate float32 actor gradients in float64 to avoid norm overflow."""
    norms = [torch.linalg.vector_norm(p.grad.detach().double()) for p in parameters if p.grad is not None]
    if not norms:
        return torch.zeros((), dtype=torch.float64)
    return torch.linalg.vector_norm(torch.stack(norms))


@torch.no_grad()
def clip_gradients(parameters, max_norm=1.0):
    """Clip finite gradients without overflowing or silently zeroing the update."""
    parameters = list(parameters)
    norm = parameter_gradient_norm(parameters)
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite global gradient norm; update was not applied")
    if norm > max_norm:
        scale = max_norm / norm
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.copy_((parameter.grad.double() * scale).to(parameter.grad.dtype))
    return norm
