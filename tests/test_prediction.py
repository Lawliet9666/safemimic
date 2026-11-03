import torch

from safety_mimic.ball import aimed_throw
from safety_mimic.geometry import CollisionGeometry


class TwoProxies(CollisionGeometry):
    def __init__(self):
        self.centers = torch.tensor([[[0.0, 0, 0], [0.0, 0, 1.4]]], dtype=torch.double, requires_grad=True)
        self.radii = torch.tensor([0.1, 0.1], dtype=torch.double)

    def centers_and_velocities(self, state):
        return self.centers, torch.zeros_like(self.centers)


def test_head_aimed_throw_does_not_mark_stationary_foot_as_threatened():
    geometry = TwoProxies()
    ball = aimed_throw(
        torch.tensor([[0.0, 0, 1.4]], dtype=torch.double), torch.Generator().manual_seed(17), azimuth=0
    )
    prediction = geometry.predictive_clearance(None, ball)
    assert prediction[0, 0] > 0.5
    assert prediction[0, 1] < 0
    derivative = torch.autograd.grad(prediction.sum(), geometry.centers, create_graph=True)[0]
    curvature = torch.autograd.grad(derivative.sum(), geometry.centers)[0]
    assert torch.isfinite(derivative).all()
    assert torch.isfinite(curvature).all()
