"""Compare full collision-proxy kinematics with independent MuJoCo FK."""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from safety_mimic.geometry import CollisionGeometry

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "checkpoints/g1-lafan1-040000/config.json").read_text())
    geometry = CollisionGeometry(
        ROOT / "vendor/supermimic/data/g1_description/g1_29dof.xml",
        config["control"]["joint_names"],
        dtype=torch.double,
    )
    model = geometry.model
    data = mujoco.MjData(model)
    generator = torch.Generator().manual_seed(17)
    max_error = 0.0
    max_velocity_error = 0.0
    for _ in range(20):
        q = geometry.q_lo + (geometry.q_hi - geometry.q_lo) * torch.rand(29, generator=generator)
        data.qpos[:3] = [0.2, -0.1, 0.8]
        quaternion = np.array([0.97, 0.1, -0.15, 0.12])
        quaternion /= np.linalg.norm(quaternion)
        data.qpos[3:7] = quaternion
        for i, name in enumerate(geometry.joint_names):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = q[i]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        world_velocity = np.array([0.2, 0.1, -0.3])
        world_omega = np.array([-0.3, 0.4, 0.2])
        qd = torch.randn(29, generator=generator).double()
        data.qvel[:3] = world_velocity
        data.qvel[3:6] = rotation.reshape(3, 3).T @ world_omega
        for i, name in enumerate(geometry.joint_names):
            data.qvel[model.jnt_dofadr[model.joint(name).id]] = qd[i]
        mujoco.mj_forward(model, data)
        state = (
            torch.tensor(data.qpos[:3])[None],
            torch.tensor(rotation.reshape(3, 3))[None],
            torch.tensor(world_velocity)[None],
            torch.tensor(world_omega)[None],
            q[None].double(),
            qd[None],
        )
        centers, velocities = geometry.centers_and_velocities(state)
        max_error = max(
            max_error, float(np.abs(centers.numpy()[0] - data.geom_xpos[geometry.geom_ids]).max())
        )
        reference_velocities = []
        for geom in geometry.geom_ids:
            spatial = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom, spatial, 0)
            reference_velocities.append(spatial[3:])
        max_velocity_error = max(
            max_velocity_error, float(np.abs(velocities.numpy()[0] - reference_velocities).max())
        )
    if max_error > 1e-8:
        raise AssertionError(f"Collision FK disagrees with MuJoCo: {max_error}")
    if max_velocity_error > 1e-8:
        raise AssertionError(f"Collision-point velocities disagree with MuJoCo: {max_velocity_error}")
    # Ensure every vertex of every collidable mesh lies inside its local sphere.
    uncovered = []
    for local_index, geom in enumerate(geometry.geom_ids):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        radius = np.linalg.norm(model.mesh_vert[start : start + count], axis=-1).max()
        if radius > float(geometry.radii[local_index]) + 1e-6:
            uncovered.append(geometry.geom_names[local_index])
    if uncovered:
        raise AssertionError(f"Uncovered collision meshes: {uncovered}")
    result = {
        "max_geom_center_error_m": max_error,
        "collision_geometries": len(geometry.geom_ids),
        "max_geom_velocity_error_m_s": max_velocity_error,
        "covered_links": geometry.covered_links,
        "all_mesh_vertices_enclosed": True,
        "scope": "geometric validation, not a collision-avoidance guarantee",
    }
    destination = ROOT / "results"
    destination.mkdir(exist_ok=True)
    (destination / "geometry_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
