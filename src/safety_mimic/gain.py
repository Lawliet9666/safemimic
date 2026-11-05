"""Bounded, observation-conditioned linear class-K coefficient and score loss."""

import math

import torch
from torch import nn


class ClassKGainHead(nn.Module):
    def __init__(self, features, hidden=32, minimum=0.05, maximum=0.8, initial=0.2):
        super().__init__()
        if not 0 < minimum < initial < maximum <= 1 or hidden < 1:
            raise ValueError("Require 0 < gain_min < gain_initial < gain_max <= 1")
        self.hidden, self.minimum, self.maximum, self.initial = hidden, minimum, maximum, initial
        fraction = (initial - minimum) / (maximum - minimum)
        self.initial_logit = math.log(fraction / (1 - fraction))
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.ELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, self.initial_logit)

    def logits(self, features):
        return self.net(features).squeeze(-1)

    def transform(self, latent):
        return self.minimum + (self.maximum - self.minimum) * latent.sigmoid()

    def forward(self, features):
        return self.transform(self.logits(features))

    def distribution(self, features, std):
        if not math.isfinite(std) or std <= 0:
            raise ValueError("Gain exploration standard deviation must be positive and finite")
        return torch.distributions.Normal(self.logits(features), std)


def discounted_returns(rewards, gamma):
    if not 0 <= gamma <= 1 or rewards.ndim != 2:
        raise ValueError("Expected time-by-environment rewards and 0 <= gamma <= 1")
    following = torch.zeros_like(rewards[-1])
    values = []
    for reward in reversed(rewards.unbind()):
        following = reward + gamma * following
        values.append(following)
    return torch.stack(values[::-1])


def gain_policy_loss(head, features, latents, old_log_prob, advantages, *, std=0.4, clip=0.2, prior=0.01):
    """PPO score-function update on pre-sigmoid samples, with no QP derivative.

    The bounded transform is fixed and invertible, so its Jacobian cancels from
    the old/new likelihood ratio. Samples and rollout returns must be detached.
    """
    if any(value.requires_grad for value in (features, latents, old_log_prob, advantages)):
        raise ValueError("Gain rollout evidence must be detached")
    if not 0 < clip < 1 or prior < 0:
        raise ValueError("Invalid PPO clip or gain prior")
    log_prob = head.distribution(features, std).log_prob(latents)
    difference = log_prob - old_log_prob
    ratio = difference.exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
    regularizer = (head.logits(features) - head.initial_logit).square().mean()
    loss = -surrogate.mean() + prior * regularizer
    with torch.no_grad():
        kl = ((ratio - 1) - difference).mean()
    return loss, {"gain_approx_kl": float(kl), "gain_prior": float(regularizer.detach())}
