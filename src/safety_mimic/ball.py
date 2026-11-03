"""Analytic projectile dynamics shared by collection and imagined rollouts."""

from dataclasses import dataclass

import torch


@dataclass
class BallState:
    position: torch.Tensor
    velocity: torch.Tensor
    radius: torch.Tensor

    def step(self, dt: float, gravity: float = -9.81) -> "BallState":
        acceleration = self.position.new_tensor([0.0, 0.0, gravity])
        return BallState(
            self.position + dt * self.velocity + 0.5 * dt**2 * acceleration,
            self.velocity + dt * acceleration,
            self.radius,
        )


def aimed_throw(
    targets: torch.Tensor,
    generator: torch.Generator,
    flight_time: float = 0.7,
    distance: float = 4.0,
    radius: float = 0.08,
    azimuth: float | None = None,
) -> BallState:
    """Generate seeded throws reaching targets [m] after flight_time [s]."""
    if flight_time <= 0 or distance <= 0 or radius <= 0:
        raise ValueError("Throw time, distance, and radius must be positive")
    count = targets.shape[0]
    angle = (
        torch.rand(count, device=targets.device, dtype=targets.dtype, generator=generator) * (2 * torch.pi)
        if azimuth is None
        else targets.new_full((count,), azimuth)
    )
    offset = torch.stack((distance * angle.cos(), distance * angle.sin(), torch.zeros_like(angle)), -1)
    position = targets + offset
    acceleration = targets.new_tensor([0.0, 0.0, -9.81])
    velocity = (targets - position - 0.5 * acceleration * flight_time**2) / flight_time
    return BallState(position, velocity, targets.new_full((count,), radius))
