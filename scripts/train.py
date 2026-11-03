"""Fine-tune the complete pretrained actor and ball branch; QP is training-only."""

import argparse
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pretrained", default="checkpoints/g1-lafan1-040000")
parser.add_argument("--world_model", default="outputs/checkpoints/world_model_broad.pt")
parser.add_argument("--output", type=Path, default=Path("outputs/runs/safetymimic_seed0"))
parser.add_argument("--method", choices=["safetymimic", "reward", "identity"], default="safetymimic")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--updates", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episode_steps", type=int, default=150)
parser.add_argument("--flight_time", type=float, default=0.7)
parser.add_argument("--throw_distance", type=float, default=4)
parser.add_argument("--exploration_noise", type=float, default=0.03)
parser.add_argument("--checkpoint_updates", default="0,100,300")
parser.add_argument("--linearization_steps", type=int, default=3)
parser.add_argument("--velocity_time", type=float, default=0.25)
parser.add_argument("--anticipation_mode", choices=["all", "threat"], default="threat")
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--residual_margin", type=float, default=0.002)
parser.add_argument("--trust_radius", type=float, default=1.0)
parser.add_argument("--correction_weight", type=float, default=1)
parser.add_argument(
    "--slack_weight", type=float, default=10, help="Training slack-loss weight, not QP penalty"
)
parser.add_argument(
    "--baseline_avoidance_weight", type=float, default=10,
    help="Geometric reward shaping for --method reward only; unused by the proposed method",
)
add_launcher_args(parser)
args = parser.parse_args()

import torch
from supermimic import networks
from supermimic.env import make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP
from supermimic.motion import MotionLib
from supermimic.networks import Critic, WorldModel
from supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list

from safety_mimic.ball import BallState
from safety_mimic.evaluation import dump_json
from safety_mimic.experiment import action_bounds, resolve_artifact
from safety_mimic.geometry import CollisionGeometry
from safety_mimic.isaac_env import DodgeballSim
from safety_mimic.layer import SafetyConfig, SafetyLayer
from safety_mimic.policy import BallConditionedPolicy, save_policy, update_history
from safety_mimic.provenance import write_provenance
from safety_mimic.runtime import configure_runtime, state_digest
from safety_mimic.training import (
    OBJECTIVE_FORMAT,
    LossConfig,
    clip_gradients,
    imagined_objective,
    parameter_gradient_norm,
    sample_replay,
    update_critic,
)


def main():
    configure_runtime(args.device)
    if args.updates < 1 or not 1 <= args.batch_size <= args.num_envs:
        raise ValueError("Need positive updates and 1 <= batch_size <= num_envs")
    if args.learning_rate <= 0 or args.exploration_noise < 0:
        raise ValueError("Invalid learning rate or exploration noise")
    if not 0 <= args.gae_lambda <= 1 or args.critic_epochs < 1 or args.critic_minibatches < 1:
        raise ValueError("Invalid critic settings")
    output = resolve_artifact(ROOT, args.output).resolve()
    if not output.is_relative_to(ROOT / "outputs/runs"):
        raise ValueError("Training output must be inside outputs/runs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh output directory; checkpoints are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    loss_config = LossConfig(**{f.name: getattr(args, f.name) for f in fields(LossConfig)})
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = None
    device = "cuda:0"
    pretrained = resolve_artifact(ROOT, args.pretrained)
    model_path = resolve_artifact(ROOT, args.world_model)
    policy = BallConditionedPolicy.from_pretrained(pretrained, device=device).train()
    model_record = torch.load(model_path, map_location=device, weights_only=True)
    world_model = WorldModel(hidden=model_record["hidden"]).to(device).eval().requires_grad_(False)
    world_model.load_state_dict(model_record["wm"], strict=True)
    initial_model_hash = state_digest(world_model)
    initial_groups = {name: state_digest(module) for name, module in policy.parameter_groups().items()}
    normalizers = {
        name: state_digest(module)
        for name, module in (("observations", policy.tracker.obs_norm), ("goals", policy.tracker.goal_norm))
    }
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(str(ROOT / "data/standing"), kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    critic_input_dim = INV_DIM + CRITIC_EXTRA + 730 + 6
    critic = Critic(critic_input_dim).to(device).train().requires_grad_(False)
    initial_critic_hash = state_digest(critic.net)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
    geometry = CollisionGeometry(
        ROOT / "vendor/supermimic/data/g1_description/g1_29dof.xml", G1_JOINT_NAMES, device=device
    )
    safety_config = SafetyConfig(
        linearization_steps=args.linearization_steps,
        velocity_time=args.velocity_time,
        anticipation_mode=args.anticipation_mode,
        alpha=args.alpha,
        residual_margin=args.residual_margin,
        trust_radius=args.trust_radius,
    )
    safety = SafetyLayer(world_model, geometry, safety_config) if args.method != "reward" else None
    config.update(
        imagined_rollouts=1,
        objective=OBJECTIVE_FORMAT,
        training_uses_qp=args.method != "reward",
        loss=asdict(loss_config),
        safety=asdict(safety_config),
        collection_uses_qp=False,
        deployment_uses_qp=False,
        terminal_critic=True,
        task_reward="original_supermimic_eight_terms",
        auxiliary_losses=["qp_correction", "qp_slack"] if safety is not None else [],
        correction_teacher="passes_affine_slack_and_nonlinear_checks",
        critic_reward="task_minus_weighted_correction_and_slack",
    )
    dump_json(output / "config.json", config)
    write_provenance(output, ROOT, [pretrained / "policy.safetensors", model_path])
    updates, transitions, imagined_transitions = 0, 0, 0
    episodes, replay = [], []
    history = None
    checkpoints = {int(value) for value in args.checkpoint_updates.split(",") if value.strip()}
    generator = torch.Generator(device=device).manual_seed(args.seed + 321)
    torch.cuda.reset_peak_memory_stats()

    def metadata():
        return {
            "updates": updates,
            "sim_transitions": transitions,
            "imagined_transitions": imagined_transitions,
            "wall_seconds": time.monotonic() - start,
            "seed": args.seed,
            "world_model_frozen": True,
            "world_model_sha256_initial": initial_model_hash,
            "initial_parameter_group_sha256": initial_groups,
            "imagined_rollouts": 1,
            "objective": OBJECTIVE_FORMAT,
            "training_uses_qp": args.method != "reward",
            "all_actor_parameters_trainable": all(p.requires_grad for p in policy.parameters()),
            "normalization_frozen": True,
            "terminal_critic": True,
            "collection_uses_qp": False,
            "deployment_uses_qp": False,
        }

    def save(name):
        save_policy(output / name, policy, config=config, training=metadata())
        critic_path = output / name.replace("policy", "critic")
        temporary = critic_path.with_suffix(".pt.tmp")
        torch.save({
            "format": "safetymimic.critic.v1",
            "input_dim": critic_input_dim,
            "critic": critic.state_dict(),
            "optimizer": critic_optimizer.state_dict(),
            "config": config,
            "training": metadata(),
        }, temporary)
        temporary.replace(critic_path)

    sim_cfg = make_sim(device, args)
    sim_cfg.dt, sim_cfg.physics.num_substeps = 0.005, 1
    try:
        with launch_simulation(sim_cfg, args):
            env = DodgeballSim(
                sim_cfg,
                args.num_envs,
                motion,
                mdp,
                geometry,
                seed=args.seed,
                episode_steps=args.episode_steps,
                flight_time=args.flight_time,
                throw_distance=args.throw_distance,
            )
            if 0 in checkpoints:
                save("policy_000000.pt")
            for _ in range(args.updates):
                with torch.no_grad():
                    state, ball = env.state(), env.ball_state()
                    observation = mdp.obs(state, env.a_prev, env.xi, env.f)
                    history = update_history(history, observation, env.ep_step == 0)
                    lower, upper = action_bounds(geometry, env)
                    raw = policy(history, motion.goal(env.f, 10), state, ball)
                    noise = args.exploration_noise * torch.randn(
                        raw.shape, device=device, generator=generator
                    )
                    action = (raw + noise).clamp(lower, upper)
                    if not bool(torch.isfinite(raw).all() & torch.isfinite(action).all()):
                        raise RuntimeError("Nonfinite nominal action; simulator was not advanced")
                    replay.append(
                        {
                            "state": tuple(value.detach().clone() for value in state),
                            "ball": BallState(
                                *(getattr(ball, key).clone() for key in ("position", "velocity", "radius"))
                            ),
                            "history": history.clone(),
                            "frame": env.f.clone(),
                            "xi": env.xi.clone(),
                            "previous": env.a_prev.clone(),
                        }
                    )
                    replay = replay[-128:]
                    transition = env.step(action)
                    transitions += args.num_envs
                    ids = transition["done"].nonzero().flatten()
                    for index in ids.tolist():
                        recovered = bool(env.state()[0][index, :2].norm() < 0.35)
                        episodes.append(
                            {
                                "episode": int(env.episode_count[index]),
                                "env": index,
                                "hit": bool(transition["hit"][index]),
                                "fall": bool(transition["fell"][index]),
                                "recovered": recovered,
                                "success": not bool(transition["hit"][index] or transition["fell"][index])
                                and recovered,
                                "execution_uses_qp": False,
                            }
                        )
                    if len(ids):
                        env.reset_episode(ids)
                sample, _ = sample_replay(replay, args.batch_size)
                # Joint calibration in the sampled replay state determines its bounds.
                sample_lower = (geometry.q_lo - env.default_q - sample["xi"][:, 1:30]) / env.alpha
                sample_upper = (geometry.q_hi - env.default_q - sample["xi"][:, 1:30]) / env.alpha
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                tick = time.monotonic()
                loss, metrics, critic_batch = imagined_objective(
                    policy,
                    world_model,
                    safety,
                    mdp,
                    motion,
                    geometry,
                    sample,
                    sample_lower.clamp_min(-5),
                    sample_upper.clamp_max(5),
                    loss_config,
                    critic=critic,
                )
                torch.cuda.synchronize()
                forward_seconds = time.monotonic() - tick
                tick = time.monotonic()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite imagined loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in policy.parameters()):
                    raise RuntimeError("Nonfinite actor gradient; update was not applied")
                group_norms = {
                    name: float(parameter_gradient_norm(module.parameters()))
                    for name, module in policy.parameter_groups().items()
                }
                gradient_norm = float(clip_gradients(policy.parameters(), 1.0))
                optimizer.step()
                torch.cuda.synchronize()
                backward_seconds = time.monotonic() - tick
                tick = time.monotonic()
                critic.requires_grad_(True)
                critic_loss = update_critic(
                    critic, critic_optimizer, critic_batch, gamma=args.gamma,
                    gae_lambda=args.gae_lambda, epochs=args.critic_epochs,
                    minibatches=args.critic_minibatches,
                )
                critic.requires_grad_(False)
                torch.cuda.synchronize()
                critic_seconds = time.monotonic() - tick
                updates += 1
                imagined_transitions += metrics.pop("imagined_transitions")
                row = {
                    **metadata(),
                    **metrics,
                    "gradient_norm": gradient_norm,
                    "parameter_group_gradient_norms": group_norms,
                    "forward_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                    "critic_seconds": critic_seconds,
                    "critic_loss": critic_loss,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                dump_json(output / "progress.json", row)
                print(json.dumps(row, allow_nan=False), flush=True)
                if updates in checkpoints:
                    save(f"policy_{updates:06d}.pt")
            save("policy.pt")
        final_groups = {name: state_digest(module) for name, module in policy.parameter_groups().items()}
        checks = {
            "world_model_unchanged": initial_model_hash == state_digest(world_model),
            "world_model_has_no_gradients": all(p.grad is None for p in world_model.parameters()),
            "critic_parameters_changed": initial_critic_hash != state_digest(critic.net),
            "parameter_groups_changed": {
                key: value != initial_groups[key] for key, value in final_groups.items()
            },
            "policy_normalizers_unchanged": normalizers
            == {
                "observations": state_digest(policy.tracker.obs_norm),
                "goals": state_digest(policy.tracker.goal_norm),
            },
        }
        dump_json(output / "checks.json", checks)
        if not checks["world_model_unchanged"] or not checks["world_model_has_no_gradients"]:
            raise RuntimeError("Frozen world-model contract was violated")
        dump_json(output / "episodes.json", episodes)
        dump_json(output / "COMPLETE.json", {**metadata(), "checks": checks})
    except BaseException as error:
        dump_json(output / "INCOMPLETE.json", {**metadata(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
