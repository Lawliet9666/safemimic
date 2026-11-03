"""Conservative per-collision-geom spheres and differentiable full-tree kinematics.

All collision geometries, including fixed bodies, are represented. Visual-only
geometries are excluded. Clearance uses spheres enclosing the simulator geoms;
actual simulator contacts remain the independent evaluation criterion.
"""

from pathlib import Path

import mujoco
import numpy as np
import torch

RobotState = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class CollisionGeometry:
    def __init__(self, path: str | Path, joint_names: list[str], device="cpu", dtype=torch.float32):
        self.model = model = mujoco.MjModel.from_xml_path(str(path))
        self.joint_names = joint_names
        self.body_names = [model.body(i).name for i in range(1, model.nbody)]
        self.parents = [int(model.body_parentid[i]) - 1 for i in range(1, model.nbody)]
        self.joints = []
        for body in range(1, model.nbody):
            ids = range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body])
            indices = []
            for joint in ids:
                if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("Only a free root and hinge joints are supported")
                if np.linalg.norm(model.jnt_pos[joint]) > 1e-8:
                    raise ValueError("Joint anchors away from the body origin require explicit support")
                indices.append((joint_names.index(model.joint(joint).name), model.jnt_axis[joint].copy()))
            self.joints.append(indices)

        def tensor(value):
            return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)

        self.offsets = tensor(model.body_pos[1:])
        matrices = []
        for quaternion in model.body_quat[1:]:
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, quaternion)
            matrices.append(matrix.reshape(3, 3))
        self.rotations = tensor(matrices)
        self.axes = [[(index, tensor(axis)) for index, axis in joints] for joints in self.joints]
        geom_ids = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] > 0 and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]
        if not geom_ids:
            raise ValueError("The robot has no collidable geometries")
        self.geom_ids = geom_ids
        self.geom_names = [model.geom(i).name or f"geom_{i}" for i in geom_ids]
        self.body_indices = torch.tensor(model.geom_bodyid[geom_ids] - 1, device=device, dtype=torch.long)
        self.local_centers = tensor(model.geom_pos[geom_ids])
        self.radii = tensor(model.geom_rbound[geom_ids])
        self.covered_links = sorted({model.body(int(model.geom_bodyid[i])).name for i in geom_ids})
        self.q_lo = tensor([model.jnt_range[model.joint(n).id, 0] for n in joint_names])
        self.q_hi = tensor([model.jnt_range[model.joint(n).id, 1] for n in joint_names])

    def poses(self, state: RobotState):
        p, rotation, velocity, omega, q, qd = state
        positions, rotations, velocities, omegas = [], [], [], []
        for i, parent in enumerate(self.parents):
            if parent < 0:
                if self.model.body_jntnum[i + 1] == 0:
                    raise ValueError("Robot must have a floating root")
                pi, ri, vi, wi = p, rotation, velocity, omega
            else:
                offset = (rotations[parent] @ self.offsets[i].unsqueeze(-1)).squeeze(-1)
                pi = positions[parent] + offset
                vi = velocities[parent] + torch.cross(omegas[parent], offset, dim=-1)
                ri = rotations[parent] @ self.rotations[i]
                wi = omegas[parent]
                for index, axis in self.axes[i]:
                    world_axis = (ri @ axis.unsqueeze(-1)).squeeze(-1)
                    wi = wi + world_axis * qd[:, index : index + 1]
                    x, y, z = axis.unbind()
                    zero = torch.zeros_like(x)
                    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
                    angle = q[:, index, None, None]
                    ri = ri @ (
                        torch.eye(3, device=q.device, dtype=q.dtype)
                        + angle.sin() * skew
                        + (1 - angle.cos()) * (skew @ skew)
                    )
            positions.append(pi)
            rotations.append(ri)
            velocities.append(vi)
            omegas.append(wi)
        return tuple(torch.stack(values, 1) for values in (positions, rotations, velocities, omegas))

    def centers_and_velocities(self, state: RobotState):
        positions, rotations, velocities, omegas = self.poses(state)
        offset = (rotations[:, self.body_indices] @ self.local_centers[..., None]).squeeze(-1)
        return (
            positions[:, self.body_indices] + offset,
            velocities[:, self.body_indices] + torch.cross(omegas[:, self.body_indices], offset, dim=-1),
        )

    def barriers(self, state: RobotState, ball, margin: float = 0.02):
        centers, velocities = self.centers_and_velocities(state)
        relative_position = centers - ball.position[:, None]
        relative_velocity = velocities - ball.velocity[:, None]
        distance = torch.linalg.vector_norm(relative_position, dim=-1).clamp_min(1e-9)
        clearance = distance - self.radii - ball.radius[:, None] - margin
        closing_rate = (relative_position * relative_velocity).sum(-1) / distance
        return clearance, closing_rate

    def predictive_clearance(
        self, state: RobotState, ball, margin=0.02, horizon=1.0, samples=21, temperature=0.02
    ):
        """Smooth lower envelope of ballistic clearances under constant link velocity.

        This feature controls an auxiliary anticipation weight. It is neither a
        dynamics rollout nor a safety certificate. Geometric inequalities remain
        enforced independently for every link.
        """
        centers, velocities = self.centers_and_velocities(state)
        times = torch.linspace(0, horizon, samples, device=centers.device, dtype=centers.dtype)
        links = centers[:, :, None] + velocities[:, :, None] * times[None, None, :, None]
        gravity = centers.new_tensor([0, 0, -9.81])
        projectile = (
            ball.position[:, None]
            + ball.velocity[:, None] * times[None, :, None]
            + 0.5 * times[None, :, None].square() * gravity
        )
        # A conservative smooth distance has finite second derivatives even
        # when a predicted center coincides exactly with the projectile.
        distances = ((links - projectile[:, None]).square().sum(-1) + 1e-8).sqrt() - 1e-4
        clearances = distances - self.radii[None, :, None] - ball.radius[:, None, None] - margin
        return -temperature * torch.logsumexp(-clearances / temperature, dim=-1)
