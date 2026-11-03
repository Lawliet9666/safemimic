"""Isaac Lab adapter with a physical ball and contact checks at every 5 ms substep.

Import only after the kit-less launcher has selected the physics backend.
"""

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from supermimic.env import TrackingSceneCfg, TrackingSim
from supermimic.mdp import flatten_state
from supermimic.rotations import mat_to_quat

from .ball import BallState, aimed_throw


@configclass
class DodgeballSceneCfg(TrackingSceneCfg):
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.08,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.0, angular_damping=0.0),
            mass_props=sim_utils.MassCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.08)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 1.0)),
    )
    ball_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"],
        track_contact_points=True,
    )


class DodgeballSim(TrackingSim):
    scene_cfg_type = DodgeballSceneCfg

    def __init__(
        self,
        sim_cfg,
        num_envs,
        motion,
        mdp,
        geometry,
        *,
        seed=0,
        episode_steps=150,
        flight_time=0.7,
        throw_distance=4.0,
        diagnostics=False,
    ):
        if abs(sim_cfg.dt - 0.005) > 1e-9 or sim_cfg.physics.num_substeps != 1:
            raise ValueError(
                "DodgeballSim requires 5 ms physics with one substep; it executes four per control"
            )
        super().__init__(sim_cfg, num_envs, motion, mdp, eval_mode=True, reset_noise=False)
        self.ball_object = self.scene["ball"]
        self.contact_sensor = self.scene["ball_contacts"]
        self.geometry = geometry
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.seed = seed
        self.episode_count = torch.full((num_envs,), -1, device=self.device, dtype=torch.long)
        self.episode_steps = episode_steps
        self.flight_time, self.throw_distance = flight_time, throw_distance
        self.diagnostics = diagnostics
        self.hit = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.fell = torch.zeros_like(self.hit)
        self.reset_episode(torch.arange(num_envs, device=self.device))

    def ball_state(self):
        pose = self.ball_object.data.root_link_pose_w.torch
        velocity = self.ball_object.data.root_link_vel_w.torch
        return BallState(
            pose[:, :3].clone() - self.scene.env_origins,
            velocity[:, :3].clone(),
            pose.new_full((self.N,), 0.08),
        )

    def reset_episode(self, ids):
        self.done_eval[ids] = False
        self.failed_eval[ids] = False
        self.reset(ids, torch.zeros(len(ids), device=self.device, dtype=torch.long))
        self.refresh()
        centers, _ = self.geometry.centers_and_velocities(self.state())
        candidate_links = {
            "head_link",
            "torso_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_knee_link",
            "right_knee_link",
        }
        candidates = [
            i
            for i, geom in enumerate(self.geometry.geom_ids)
            if self.geometry.model.body(int(self.geometry.model.geom_bodyid[geom])).name in candidate_links
        ]
        if not candidates:
            raise ValueError("No body regions available for targeted throws")
        # Each (environment, episode) has its own seed. Different failure/reset
        # timings cannot change the held-out throw assigned to another episode.
        self.episode_count[ids] += 1
        balls = []
        for env_id in ids.tolist():
            throw_id = int(self.episode_count[env_id]) * self.N + env_id
            self.generator.manual_seed(self.seed + 104729 * throw_id)
            selected = candidates[
                int(torch.randint(len(candidates), (), device=self.device, generator=self.generator))
            ]
            balls.append(
                aimed_throw(
                    centers[env_id : env_id + 1, selected],
                    self.generator,
                    flight_time=self.flight_time,
                    distance=self.throw_distance,
                )
            )
        ball = BallState(
            *(
                torch.cat([getattr(value, key) for value in balls], 0)
                for key in ("position", "velocity", "radius")
            )
        )
        pose = torch.cat(
            (
                ball.position + self.scene.env_origins[ids],
                ball.position.new_tensor([0, 0, 0, 1]).expand(len(ids), -1),
            ),
            -1,
        )
        self.ball_object.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.ball_object.write_root_velocity_to_sim_index(
            root_velocity=torch.cat((ball.velocity, torch.zeros_like(ball.velocity)), -1), env_ids=ids
        )
        self.hit[ids] = False
        self.fell[ids] = False
        self.scene.reset(ids)
        self.refresh()

    def load_throws(self, rows, initial):
        """Restore the manifest state, independent of policy RNG and batch layout."""
        ids = torch.arange(len(rows), device=self.device)
        self.reset_episode(ids)
        state = tuple(
            torch.tensor(v, device=self.device).float()[None].expand(len(rows), *torch.tensor(v).shape)
            for v in initial["state"]
        )
        p, rotation, velocity, omega, q, qd = state
        quat = mat_to_quat(rotation)
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat((p + self.scene.env_origins[ids], quat[:, 1:], quat[:, :1]), -1), env_ids=ids
        )
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=torch.cat((velocity, omega), -1), env_ids=ids
        )
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=ids)
        self.robot.set_joint_position_target_index(target=q, env_ids=ids)
        self.xi[ids] = torch.tensor(initial["xi"], device=self.device)
        self.f[ids] = initial["frame"]
        self.a_prev[ids] = torch.tensor(initial["previous_action"], device=self.device)
        self._apply_friction()
        position = torch.tensor([r["ball_position"] for r in rows], device=self.device)
        ball_quat = position.new_tensor([0, 0, 0, 1]).expand(len(rows), -1)
        self.ball_object.write_root_pose_to_sim_index(
            root_pose=torch.cat((position + self.scene.env_origins[ids], ball_quat), -1), env_ids=ids
        )
        velocity = torch.tensor([r["ball_velocity"] for r in rows], device=self.device)
        self.ball_object.write_root_velocity_to_sim_index(
            root_velocity=torch.cat((velocity, torch.zeros_like(velocity)), -1), env_ids=ids
        )
        self.scene.reset(ids)
        self.refresh()
        # Fail before comparison if a backend silently changes the prescribed state.
        error = max(float((a - b).abs().max()) for a, b in zip(self.state(), state))
        if error > 2e-5:
            raise RuntimeError(f"Manifest reset mismatch: {error}")

    def step(self, action):
        before = self.state()
        ball_before = self.ball_state()
        frame = self.f.clone()
        target = self.default_q + self.xi[:, 1:30] + self.alpha * action
        self.robot.set_joint_position_target_index(target=target)
        self.scene.write_data_to_sim()
        hit_step = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        contact_events = [[] for _ in range(self.N)]
        substep_min = torch.full((self.N,), float("inf"), device=self.device)
        # Four independent physics steps preserve the 50 Hz control interface
        # while observing ball-to-any-link contact between control samples.
        for substep in range(4):
            self.sim.step(render=False)
            self.scene.update(0.005)
            matrix = self.contact_sensor.data.normal_force_matrix_w.torch
            hit_step |= torch.linalg.vector_norm(matrix, dim=-1).flatten(1).amax(-1) > 1e-5
            if self.diagnostics:
                forces = torch.linalg.vector_norm(matrix, dim=-1).amax(1)
                for env_id, column in (forces > 1e-5).nonzero().tolist():
                    contact_events[env_id].append(
                        {
                            "time": float(self.ep_step[env_id]) * 0.02 + (substep + 1) * 0.005,
                            "link": self.contact_sensor.filter_object_names[column],
                            "force_n": float(forces[env_id, column]),
                        }
                    )
                clearance, _ = self.geometry.barriers(self.state(), self.ball_state())
                substep_min = torch.minimum(substep_min, clearance.amin(-1))
        following = self.state()
        ball_following = self.ball_state()
        self.f = torch.minimum(frame + 1, self.motion.clip_end[0] - 1)
        reward = self.mdp.reward(following, self.f, action, self.a_prev)
        fall = self.mdp.termination(following, self.f)
        self.hit |= hit_step
        self.fell |= fall
        self.ep_step += 1
        done = self.hit | self.fell | (self.ep_step >= self.episode_steps)
        result = {
            "s": flatten_state(before),
            "s1": flatten_state(following),
            "a": action.clone(),
            "xi": self.xi.clone(),
            "f": frame,
            "r": reward,
            "done": done.clone(),
            "hit": self.hit.clone(),
            "fell": self.fell.clone(),
            "ball": ball_before,
            "ball1": ball_following,
            "a_prev": self.a_prev.clone(),
            "contact_events": contact_events,
            "substep_min_clearance": substep_min,
        }
        self.a_prev = action.clone()
        return result

    def snapshot(self):
        state = self.state()
        return {
            "root_p": state[0].cpu(),
            "root_quat": mat_to_quat(state[1]).cpu(),
            "q": state[4].cpu(),
            "ball_p": self.ball_state().position.cpu(),
            "f": self.f.cpu(),
        }
