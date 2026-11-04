"""Joint actor/ball fine-tuning through training-only QPs and a frozen model."""

import argparse
import json
import os
import random
import signal
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from safety_mimic.settings import WORLD_MODEL_PATH, add_safety_arguments

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pretrained", default="checkpoints/g1-lafan1-040000")
parser.add_argument("--world_model", default=WORLD_MODEL_PATH)
parser.add_argument("--output", type=Path)
parser.add_argument("--resume", type=Path)
parser.add_argument("--runtime", choices=["local", "slurm"], default="local")
parser.add_argument("--cpu_threads", type=int)
parser.add_argument("--method", choices=["safetymimic", "reward", "identity"], default="safetymimic")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--updates", type=int, help="Total optimizer updates, including restored updates")
parser.add_argument("--batch_size", type=int, default=4, help="Microbatch per GPU")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--actor_lr", type=float, default=1e-5)
parser.add_argument("--ball_lr", type=float, default=5e-5)
parser.add_argument("--critic_lr", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episode_steps", type=int, default=150)
parser.add_argument("--flight_time", type=float, default=0.7)
parser.add_argument("--throw_distance", type=float, default=4)
parser.add_argument("--exploration_noise", type=float, default=0.03)
parser.add_argument("--warmup_steps", type=int, default=150)
parser.add_argument("--replay_steps", type=int, default=1000)
parser.add_argument("--collection_steps", type=int, default=1)
parser.add_argument("--checkpoint_updates", default="0,100,300,1000,3000")
parser.add_argument("--checkpoint_every", type=int, default=100)
parser.add_argument("--stop_file", type=Path)
add_safety_arguments(parser)
parser.add_argument("--correction_weight", type=float, default=1)
parser.add_argument("--slack_weight", type=float, default=10, help="Auxiliary loss weight; QP penalty stays 1e4")
parser.add_argument("--baseline_avoidance_weight", type=float, default=10)
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np
import torch
from supermimic import networks
from supermimic.env import make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP
from supermimic.motion import MotionLib
from supermimic.networks import Critic, WorldModel
from supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list

from safety_mimic.ball import BallState
from safety_mimic.checkpoint import capture_rng, device_tree, load_snapshot, restore_rng, save_snapshot
from safety_mimic.distributed import Distribution
from safety_mimic.evaluation import dump_json
from safety_mimic.experiment import action_bounds, resolve_artifact
from safety_mimic.geometry import CollisionGeometry
from safety_mimic.isaac_env import DodgeballSim
from safety_mimic.layer import SafetyConfig, SafetyLayer
from safety_mimic.policy import BallConditionedPolicy, save_policy, update_history
from safety_mimic.provenance import write_provenance
from safety_mimic.runtime import configure_runtime, state_digest
from safety_mimic.training import (
    OBJECTIVE_FORMAT, LossConfig, actor_optimizer, clip_gradients, combine_critic_batches,
    imagined_objective, parameter_gradient_norm, sample_replay, update_critic,
)


def main():
    restored = local_record = manifest = None
    if args.resume:
        restored, local_record, manifest = load_snapshot(
            resolve_artifact(ROOT, args.resume), rank=int(os.environ.get("RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
        )
        if restored["config"]["objective"] != OBJECTIVE_FORMAT:
            raise ValueError("Checkpoint training objective differs from the current trainer")
        latest = manifest.parents[2] / "checkpoints/latest.json"
        if latest.exists() and (latest.parent / json.loads(latest.read_text())["checkpoint"]).resolve() != manifest:
            raise ValueError("Resume the published latest checkpoint; rollback would duplicate run metrics")
        operational = {"resume", "output", "updates", "runtime", "cpu_threads", "device", "stop_file"}
        explicit = {word.split("=")[0] for word in sys.argv[1:] if word.startswith("--")}
        for action in parser._actions:
            key = action.dest
            if key in operational or key not in restored["config"]:
                continue
            value = restored["config"][key]
            if any(option in explicit for option in action.option_strings) and getattr(args, key) != value:
                raise ValueError(f"Resume setting differs from checkpoint: {key}")
            setattr(args, key, value)
        args.output = args.output or manifest.parents[2]
        args.updates = args.updates if args.updates is not None else restored["config"]["updates"]
    args.updates = 300 if args.updates is None else args.updates
    args.output = args.output or Path(f"outputs/runs/safetymimic_seed{args.seed}")
    device = configure_runtime(args.device, profile=args.runtime, cpu_threads=args.cpu_threads)
    args.device = device
    distribution = Distribution(device)
    output = resolve_artifact(ROOT, args.output).resolve()
    error = None
    try:
        if not output.is_relative_to(ROOT / "outputs/runs"):
            raise ValueError("Training output must be inside outputs/runs")
        if restored and output != manifest.parents[2]:
            raise ValueError("Resume in the original run directory")
        if restored and args.updates < restored["update"]:
            raise ValueError("Target updates precede the restored checkpoint")
        if not restored and output.exists() and any(output.iterdir()):
            raise ValueError("Choose a fresh run directory")
        if not 1 <= args.batch_size <= args.num_envs or args.updates < 1:
            raise ValueError("Need positive updates and 1 <= microbatch <= num_envs")
        if min(args.gradient_accumulation_steps, args.replay_steps, args.collection_steps) < 1:
            raise ValueError("Accumulation, replay and collection counts must be positive")
        if args.warmup_steps < 0 or args.checkpoint_every < 0 or args.exploration_noise < 0:
            raise ValueError("Invalid warmup/checkpoint/noise setting")
        if not np.isfinite(args.critic_lr) or args.critic_lr <= 0:
            raise ValueError("Invalid critic learning rate")
        if not 0 <= args.gae_lambda <= 1 or min(args.critic_epochs, args.critic_minibatches) < 1:
            raise ValueError("Invalid critic settings")
        if distribution.primary:
            output.mkdir(parents=True, exist_ok=True)
    except Exception as caught:
        error = caught
    distribution.agree(error)
    distribution.barrier()
    start = time.monotonic()
    wall_offset = restored["counters"]["wall_seconds"] if restored else 0
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    loss_config = LossConfig(**{f.name: getattr(args, f.name) for f in fields(LossConfig)})
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = None
    pretrained = resolve_artifact(ROOT, args.pretrained)
    model_path = resolve_artifact(ROOT, args.world_model)
    policy = BallConditionedPolicy.from_pretrained(pretrained, device=device).train()
    model_record = torch.load(model_path, map_location=device, weights_only=True)
    world_model = WorldModel(hidden=model_record["hidden"]).to(device).eval().requires_grad_(False)
    world_model.load_state_dict(model_record["wm"], strict=True)
    initial_model_hash = state_digest(world_model)
    if restored and restored["world_model_hash"] != initial_model_hash:
        raise ValueError("World-model weights differ from the resumed run")
    initial_groups = restored["initial_groups"] if restored else {
        name: state_digest(module) for name, module in policy.parameter_groups().items()
    }
    normalizers = {name: state_digest(module) for name, module in (
        ("observations", policy.tracker.obs_norm), ("goals", policy.tracker.goal_norm)
    )}
    optimizer = actor_optimizer(policy, args.actor_lr, args.ball_lr)
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(str(ROOT / "data/standing"), kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    critic = Critic(INV_DIM + CRITIC_EXTRA + 730 + 6).to(device).train().requires_grad_(False)
    initial_critic_hash = restored["initial_critic_hash"] if restored else state_digest(critic.net)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    if restored:
        policy.load_state_dict(restored["policy"], strict=True)
        critic.load_state_dict(restored["critic"], strict=True)
        optimizer.load_state_dict(restored["actor_optimizer"])
        critic_optimizer.load_state_dict(restored["critic_optimizer"])
    distribution.broadcast_module(policy)
    distribution.broadcast_module(critic)
    geometry = CollisionGeometry(
        ROOT / "vendor/supermimic/data/g1_description/g1_29dof.xml", G1_JOINT_NAMES, device=device
    )
    safety_config = SafetyConfig(**{key: getattr(args, key) for key in (
        "linearization_steps", "step_trust_radius", "backtracking_steps", "velocity_time",
        "anticipation_mode", "alpha", "residual_margin", "trust_radius",
    )})
    safety = SafetyLayer(world_model, geometry, safety_config) if args.method != "reward" else None
    config.update(
        imagined_rollouts=1, objective=OBJECTIVE_FORMAT, training_uses_qp=safety is not None,
        loss=asdict(loss_config), safety=asdict(safety_config), collection_uses_qp=False,
        deployment_uses_qp=False, terminal_critic=True, task_reward="original_supermimic_eight_terms",
        auxiliary_losses=["qp_correction", "qp_slack"] if safety is not None else [],
        correction_teacher="passes_affine_slack_and_nonlinear_checks",
        critic_reward="task_minus_weighted_correction_and_slack", world_size=distribution.world_size,
        global_batch_size=args.batch_size * args.gradient_accumulation_steps * distribution.world_size,
        simulator_resume="reset_unfinished_episodes_and_record_censoring",
    )
    updates = restored["update"] if restored else 0
    transitions = restored["counters"]["sim_transitions"] if restored else 0
    imagined_transitions = restored["counters"]["imagined_transitions"] if restored else 0
    checkpoint_sequence = restored["checkpoint_sequence"] if restored else 0
    episodes = local_record["episodes"] if restored else []
    replay = device_tree(local_record["replay"], device) if restored else []
    history = None
    checkpoints = {int(value) for value in args.checkpoint_updates.split(",") if value.strip()}
    data_seed = args.seed + 10000019 * distribution.rank
    torch.manual_seed(data_seed)
    random.seed(data_seed)
    np.random.seed(data_seed % 2**32)
    generator = torch.Generator(device=device).manual_seed(data_seed + 321)
    torch.cuda.reset_peak_memory_stats(device)
    stopped = False
    stop_file = resolve_artifact(ROOT, args.stop_file) if args.stop_file else output / "STOP_REQUESTED"
    initial_stop_stamp = stop_file.stat().st_mtime_ns if stop_file.exists() else None
    signal_received = False

    def request_stop(signum, frame):
        nonlocal signal_received
        signal_received = True

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)

    def stop_requested():
        stamp = stop_file.stat().st_mtime_ns if stop_file.exists() else None
        return distribution.any(signal_received or stamp != initial_stop_stamp)

    def metadata():
        return {
            "updates": updates, "sim_transitions": transitions, "imagined_transitions": imagined_transitions,
            "wall_seconds": wall_offset + time.monotonic() - start, "seed": args.seed,
            "world_size": distribution.world_size, "global_batch_size": config["global_batch_size"],
            "world_model_frozen": True, "world_model_sha256_initial": initial_model_hash,
            "initial_parameter_group_sha256": initial_groups, "imagined_rollouts": 1,
            "objective": OBJECTIVE_FORMAT, "training_uses_qp": safety is not None,
            "all_actor_parameters_trainable": all(p.requires_grad for p in policy.parameters()),
            "normalization_frozen": True, "terminal_critic": True,
            "collection_uses_qp": False, "deployment_uses_qp": False,
        }

    def primary_io(operation):
        caught = None
        if distribution.primary:
            try:
                operation()
            except Exception as failure:
                caught = failure
        distribution.agree(caught)

    last_saved_update = -1

    def save(env, *, final=False):
        nonlocal checkpoint_sequence, last_saved_update
        primary_io(lambda: save_policy(
            output / ("policy.pt" if final else f"policy_{updates:06d}.pt"),
            policy, config=config, training=metadata(),
        ))
        if updates == last_saved_update:
            return
        checkpoint_sequence += 1
        save_snapshot(output, updates, {
            "policy": policy.state_dict(), "critic": critic.state_dict(),
            "actor_optimizer": optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(),
            "config": config, "counters": metadata(), "world_model_hash": initial_model_hash,
            "initial_groups": initial_groups, "initial_critic_hash": initial_critic_hash,
            "checkpoint_sequence": checkpoint_sequence,
        }, {
            "replay": replay, "rng": capture_rng(generator, device), "episodes": episodes,
            "episode_count": env.episode_count.clone(), "unfinished_steps": env.ep_step.clone(),
            "history": history,
        }, distribution)
        last_saved_update = updates

    def collect(env, count):
        nonlocal history, transitions, replay
        with torch.no_grad():
            for _ in range(count):
                state, ball = env.state(), env.ball_state()
                observation = mdp.obs(state, env.a_prev, env.xi, env.f)
                history = update_history(history, observation, env.ep_step == 0)
                lower, upper = action_bounds(geometry, env)
                raw = policy(history, motion.goal(env.f, 10), state, ball)
                noise = args.exploration_noise * torch.randn(raw.shape, device=device, generator=generator)
                action = (raw + noise).clamp(lower, upper)
                if not bool(torch.isfinite(raw).all() & torch.isfinite(action).all()):
                    raise RuntimeError("Nonfinite nominal action; simulator was not advanced")
                replay.append({
                    "state": tuple(value.clone() for value in state),
                    "ball": BallState(*(getattr(ball, key).clone() for key in ("position", "velocity", "radius"))),
                    "history": history.clone(), "frame": env.f.clone(), "xi": env.xi.clone(),
                    "previous": env.a_prev.clone(),
                })
                replay = replay[-args.replay_steps:]
                transition = env.step(action)
                transitions += args.num_envs * distribution.world_size
                ids = transition["done"].nonzero().flatten()
                for index in ids.tolist():
                    recovered = bool(env.state()[0][index, :2].norm() < 0.35)
                    episodes.append({
                        "episode": int(env.episode_count[index]), "env": index, "rank": distribution.rank,
                        "hit": bool(transition["hit"][index]), "fall": bool(transition["fell"][index]),
                        "recovered": recovered, "success": not bool(
                            transition["hit"][index] or transition["fell"][index]
                        ) and recovered, "execution_uses_qp": False, "censored": False,
                    })
                if len(ids):
                    env.reset_episode(ids)

    sim_cfg = make_sim(device, args)
    sim_cfg.dt, sim_cfg.physics.num_substeps = 0.005, 1
    try:
        primary_io(lambda: dump_json(output / "STATUS.json", {"status": "running", **metadata()}))
        primary_io(lambda: dump_json(output / ("resume_config.json" if restored else "config.json"), config))
        write_provenance(output, ROOT, [pretrained / "policy.safetensors", model_path],
                         filename=f"provenance_rank{distribution.rank}_{time.time_ns()}.json")
        with launch_simulation(sim_cfg, args):
            env = DodgeballSim(
                sim_cfg, args.num_envs, motion, mdp, geometry, seed=data_seed,
                episode_steps=args.episode_steps, flight_time=args.flight_time, throw_distance=args.throw_distance,
            )
            if restored:
                for index, steps in enumerate(local_record["unfinished_steps"].tolist()):
                    if steps:
                        episodes.append({"env": index, "rank": distribution.rank, "censored": True,
                                         "episode": int(local_record["episode_count"][index]),
                                         "reason": "simulator_reset_on_resume", "observed_steps": steps})
                env.episode_count.copy_(local_record["episode_count"].to(device))
                env.reset_episode(torch.arange(args.num_envs, device=device))
                restore_rng(local_record["rng"], generator, device)
            else:
                caught = None
                try:
                    collect(env, args.warmup_steps)
                except Exception as failure:
                    caught = failure
                distribution.agree(caught)
                if 0 in checkpoints:
                    save(env)
            while updates < args.updates:
                if stop_requested():
                    stopped = True
                    break
                tick_update = time.monotonic()
                caught = None
                try:
                    collect(env, args.collection_steps)
                    optimizer.zero_grad(set_to_none=True)
                    rows, batches = [], []
                    forward_seconds = backward_seconds = 0.0
                    for _ in range(args.gradient_accumulation_steps):
                        sample, _ = sample_replay(replay, args.batch_size)
                        lower = (geometry.q_lo - env.default_q - sample["xi"][:, 1:30]) / env.alpha
                        upper = (geometry.q_hi - env.default_q - sample["xi"][:, 1:30]) / env.alpha
                        torch.cuda.synchronize(device)
                        tick = time.monotonic()
                        loss, row, batch = imagined_objective(
                            policy, world_model, safety, mdp, motion, geometry, sample,
                            lower.clamp_min(-5), upper.clamp_max(5), loss_config, critic=critic,
                        )
                        torch.cuda.synchronize(device)
                        forward_seconds += time.monotonic() - tick
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("Nonfinite imagined loss; update rejected")
                        tick = time.monotonic()
                        (loss / args.gradient_accumulation_steps).backward()
                        torch.cuda.synchronize(device)
                        backward_seconds += time.monotonic() - tick
                        rows.append(row)
                        batches.append(batch)
                except Exception as failure:
                    caught = failure
                distribution.agree(caught)
                distribution.average_gradients(policy.parameters())
                group_norms = {name: float(parameter_gradient_norm(module.parameters()))
                               for name, module in policy.parameter_groups().items()}
                gradient_norm = float(clip_gradients(policy.parameters(), 1.0))
                optimizer.step()
                tick = time.monotonic()
                critic.requires_grad_(True)
                critic_loss = update_critic(
                    critic, critic_optimizer, combine_critic_batches(batches), gamma=args.gamma,
                    gae_lambda=args.gae_lambda, epochs=args.critic_epochs, minibatches=args.critic_minibatches,
                    distribution=distribution,
                )
                critic.requires_grad_(False)
                torch.cuda.synchronize(device)
                critic_seconds = time.monotonic() - tick
                updates += 1
                imagined_transitions += sum(row.pop("imagined_transitions") for row in rows) * distribution.world_size
                metrics = distribution.mean_metrics({key: sum(row[key] for row in rows) / len(rows) for key in rows[0]})
                timing = distribution.max_metrics({"forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
                                                   "critic_seconds": critic_seconds,
                                                   "update_seconds": time.monotonic() - tick_update,
                                                   "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
                row = {**metadata(), **metrics, **timing, "gradient_norm": gradient_norm,
                       "parameter_group_gradient_norms": group_norms,
                       "critic_loss": distribution.mean_metrics({"loss": critic_loss})["loss"],
                       "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                       "critic_lr": critic_optimizer.param_groups[0]["lr"]}
                def record():
                    with (output / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    dump_json(output / "progress.json", row)
                    print(json.dumps(row, allow_nan=False), flush=True)
                primary_io(record)
                if updates in checkpoints or (args.checkpoint_every and updates % args.checkpoint_every == 0):
                    save(env)
            save(env, final=True)
        final_groups = {name: state_digest(module) for name, module in policy.parameter_groups().items()}
        checks = {
            "world_model_unchanged": initial_model_hash == state_digest(world_model),
            "world_model_has_no_gradients": all(p.grad is None for p in world_model.parameters()),
            "critic_parameters_changed": initial_critic_hash != state_digest(critic.net),
            "parameter_groups_changed": {key: value != initial_groups[key] for key, value in final_groups.items()},
            "policy_normalizers_unchanged": normalizers == {name: state_digest(module) for name, module in (
                ("observations", policy.tracker.obs_norm), ("goals", policy.tracker.goal_norm))},
        }
        distribution.agree(None if checks["world_model_unchanged"] and checks["world_model_has_no_gradients"]
                           else RuntimeError("Frozen world-model contract was violated"))
        distribution.assert_identical({"policy": state_digest(policy), "critic": state_digest(critic)})
        dump_json(output / f"episodes_rank{distribution.rank}.json", episodes)
        primary_io(lambda: dump_json(output / "checks.json", checks))
        primary_io(lambda: dump_json(output / ("STOPPED.json" if stopped else "COMPLETE.json"),
                                     {**metadata(), "checks": checks, "resumed_from": str(manifest) if manifest else None}))
        primary_io(lambda: dump_json(output / "STATUS.json", {"status": "stopped" if stopped else "complete", **metadata()}))
    except BaseException as failure:
        dump_json(output / f"INCOMPLETE_rank{distribution.rank}.json", {**metadata(), "error": repr(failure)})
        if distribution.primary:
            dump_json(output / "STATUS.json", {"status": "incomplete", **metadata(), "error": repr(failure)})
        raise
    finally:
        distribution.close()


if __name__ == "__main__":
    main()
