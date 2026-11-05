"""Train only the bounded CBF gain head from actual filtered simulator rollouts.

The nominal actor and world model are frozen. PPO likelihood ratios use sampled
pre-sigmoid gains; no actor or gain gradient passes through the QP or simulator.
This bounded pilot uses complete episodes and Monte Carlo advantages, no critic.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from safety_mimic.tracking import RunTracker, add_tracking_arguments

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runtime", choices=["local", "slurm"], default="local")
parser.add_argument("--cpu_threads", type=int)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--updates_per_round", type=int, default=25)
parser.add_argument("--minibatch", type=int, default=256)
parser.add_argument("--train_offset", type=int, default=8, help="Keep the first eight evaluation throws out of training")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--std", type=float, default=0.4, help="Exploration in pre-sigmoid gain coordinates")
parser.add_argument("--gamma", type=float, default=0.99)
add_tracking_arguments(parser)
add_launcher_args(parser)
args = parser.parse_args()

import torch
from supermimic.env import make_sim
from supermimic.mdp import normalize_xi

from safety_mimic.checkpoint import atomic_torch_save
from safety_mimic.evaluation import dump_json, sha256, validate_manifest
from safety_mimic.experiment import Controller, resolve_artifact
from safety_mimic.gain import discounted_returns, gain_policy_loss
from safety_mimic.isaac_env import DodgeballSim
from safety_mimic.policy import save_policy, update_history
from safety_mimic.provenance import write_provenance
from safety_mimic.runtime import configure_runtime, state_digest
from safety_mimic.training import clip_gradients


def main():
    device = configure_runtime(args.device, profile=args.runtime, cpu_threads=args.cpu_threads)
    if min(args.rounds, args.updates_per_round, args.minibatch) < 1 or args.num_envs < 2 or args.train_offset < 8:
        raise ValueError("Positive pilot settings, >=2 environments and training offset >=8 are required")
    if not 0 < args.learning_rate < 1 or not 0 <= args.gamma <= 1:
        raise ValueError("Invalid learning rate or discount")
    output = resolve_artifact(ROOT, args.output).resolve()
    if not output.is_relative_to(ROOT / "outputs/runs") or output.exists():
        raise ValueError("Choose a fresh output directory inside outputs/runs")
    manifest_path = resolve_artifact(ROOT, args.manifest)
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    stop = args.train_offset + args.rounds * args.num_envs
    training_throws = manifest["throws"][args.train_offset:stop]
    if len(training_throws) != args.rounds * args.num_envs:
        raise ValueError("Insufficient distinct training throws; evaluation throws are never recycled")
    torch.manual_seed(args.seed)
    controller = Controller(ROOT, seed=args.seed, execution_filter=True, device=device)
    policy, safety = controller.policy, controller.safety
    policy.enable_gain_head()
    policy.gain_head.requires_grad_(True)
    head = policy.gain_head
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=device).manual_seed(args.seed + 121)
    model = safety.dynamics
    initial_nominal = {key: state_digest(module) for key, module in policy.parameter_groups().items() if key != "gain_head"}
    initial_model, initial_head = state_digest(model), state_digest(head)
    normalizers = {key: state_digest(module) for key, module in (
        ("observation", policy.tracker.obs_norm), ("goal", policy.tracker.goal_norm))}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(method="filter", adaptive_gain=True, objective="safetymimic.filter_gain_score.v1",
                  world_model=str(controller.world_model_path), safety=asdict(safety.config),
                  collection_uses_qp=True, deployment_uses_qp=True, imagined_uses_qp=False,
                  qp_training_gradient=False, nominal_actor_frozen=True, world_model_frozen=True,
                  gain_bounds=[head.minimum, head.maximum], gain_initial=head.initial,
                  manifest_sha256=sha256(manifest_path), training_throw_ids=[t["throw_id"] for t in training_throws],
                  evaluation_throw_ids=[t["throw_id"] for t in manifest["throws"][:8]],
                  reward={"tracking": "original eight-term reward", "first_contact": -10,
                          "first_fall": -20, "failed_filter_action": -0.02, "successful_episode": 5},
                  sampling=f"{args.rounds} complete rollout batches, with fresh on-policy data between PPO batches",
                  advantage="Monte Carlo return-to-go minus leave-one-out environment mean at each time")
    output.mkdir(parents=True)
    dump_json(output / "config.json", config)
    write_provenance(output, ROOT, [resolve_artifact(ROOT, "checkpoints/g1-lafan1-040000/policy.safetensors"),
                                   controller.world_model_path, manifest_path])
    start, updates = time.monotonic(), 0

    def metadata(status):
        return {"status": status, "updates": updates, "wall_seconds": time.monotonic() - start,
                "objective": config["objective"], "qp_training_gradient": False,
                "world_model_sha256_initial": initial_model, "nominal_actor_frozen": True}

    def save():
        save_policy(output / f"policy_{updates:06d}.pt", policy, config=config, training=metadata("checkpoint"))
        atomic_torch_save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                           "config": config, "updates": updates, "rng": generator.get_state(),
                           "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device)},
                          output / "training_state.pt")

    save()
    dump_json(output / "STATUS.json", metadata("running"))
    sim_cfg = make_sim(device, args)
    sim_cfg.dt, sim_cfg.physics.num_substeps = 0.005, 1
    steps = round(manifest["episode_seconds"] / manifest["control_dt"])
    if abs(steps * 0.02 - manifest["episode_seconds"]) > 1e-8:
        raise ValueError("Pilot requires the established 50 Hz control interface")
    tracker = None
    try:
        tracker = RunTracker.from_args(output, config, args)
        with launch_simulation(sim_cfg, args):
            env = DodgeballSim(sim_cfg, args.num_envs, controller.motion, controller.mdp, controller.geometry,
                               seed=args.seed, diagnostics=False)
            for round_id in range(args.rounds):
                batch = training_throws[round_id * args.num_envs:(round_id + 1) * args.num_envs]
                env.load_throws(batch, manifest["initial"])
                history, sampled, rewards = None, [], []
                failed_count, intervention = 0, 0.0
                seen_hit = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                seen_fall = torch.zeros_like(seen_hit)
                tick = time.monotonic()
                with torch.no_grad():
                    for _ in range(steps):
                        state, ball = env.state(), env.ball_state()
                        observation = controller.mdp.obs(state, env.a_prev, env.xi, env.f)
                        history = update_history(history, observation)
                        goal = controller.motion.goal(env.f, 10)
                        nominal = policy(history, goal, state, ball)
                        features = policy.gain_features(history, goal, state, ball)
                        distribution = head.distribution(features, args.std)
                        latent = distribution.loc + args.std * torch.randn(args.num_envs, device=device, generator=generator)
                        gain = head.transform(latent)
                        lower, upper = controller.bounds(env)
                        nominal = nominal.clamp(lower, upper)
                        filtered = safety.filter(state, ball, nominal, normalize_xi(env.xi), lower, upper, gain=gain)
                        if filtered.qp.action.requires_grad or filtered.qp.slack.requires_grad:
                            raise RuntimeError("A QP graph escaped the external-filter boundary")
                        transition = env.step(filtered.qp.action)
                        reward = (transition["r"] - 10 * (transition["hit"] & ~seen_hit).float()
                                  - 20 * (transition["fell"] & ~seen_fall).float() - 0.02 * filtered.failed.float())
                        seen_hit |= transition["hit"]
                        seen_fall |= transition["fell"]
                        sampled.append((features.clone(), latent.clone(), distribution.log_prob(latent).clone(), gain.clone()))
                        rewards.append(reward)
                        failed_count += int(filtered.failed.sum())
                        intervention += float((filtered.qp.action - nominal).norm(dim=-1).sum())
                    displacement = (env.state()[0][:, :2] - state[0].new_tensor(manifest["initial"]["state"][0][:2])).norm(dim=-1)
                    successful = ~seen_hit & ~seen_fall & (displacement < 0.35)
                    rewards[-1] += 5 * successful.float()
                    reward_tensor = torch.stack(rewards)
                    returns = discounted_returns(reward_tensor, args.gamma)
                    baseline = (returns.sum(1, keepdim=True) - returns) / (args.num_envs - 1)
                    advantages = returns - baseline
                    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
                    features, latents, old_log_prob, rollout_gains = [torch.stack(x) for x in zip(*sampled)]
                    features = features.flatten(0, 1)
                    latents, old_log_prob, advantages = (x.flatten() for x in (latents, old_log_prob, advantages))
                rollout_seconds = time.monotonic() - tick
                rollout = {"round": round_id + 1, "throw_ids": [t["throw_id"] for t in batch],
                           "contacts": int(seen_hit.sum()), "falls": int(seen_fall.sum()),
                           "successful_dodges": int(successful.sum()), "filter_failures": failed_count,
                           "filter_actions": steps * args.num_envs, "rollout_seconds": rollout_seconds,
                           "mean_return": float(reward_tensor.sum(0).mean()),
                           "gain_mean": float(rollout_gains.mean()), "gain_min": float(rollout_gains.min()),
                           "gain_max": float(rollout_gains.max()), "mean_intervention": intervention / (steps * args.num_envs)}
                with (output / "rollouts.jsonl").open("a") as stream:
                    stream.write(json.dumps(rollout, allow_nan=False) + "\n")
                print("ROLLOUT " + json.dumps(rollout), flush=True)
                tracker.log_rollout(rollout, updates)
                for _ in range(args.updates_per_round):
                    indices = torch.randperm(len(latents), device=device, generator=generator)[:args.minibatch]
                    tick_update = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    loss, diagnostics = gain_policy_loss(head, features[indices], latents[indices],
                                                         old_log_prob[indices], advantages[indices], std=args.std)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite gain-head objective")
                    loss.backward()
                    gradient = float(clip_gradients(head.parameters(), 1))
                    optimizer.step()
                    updates += 1
                    row = {**metadata("running"), **diagnostics, "round": round_id + 1,
                           "loss": float(loss.detach()), "gradient_norm": gradient,
                           "update_seconds": time.monotonic() - tick_update,
                           "deterministic_gain_mean": float(head(features).mean().detach())}
                    with (output / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    dump_json(output / "progress.json", row)
                    tracker.log_training(row)
                save()
                dump_json(output / "STATUS.json", metadata("running"))
        checks = {
            "nominal_policy_unchanged": initial_nominal == {key: state_digest(module) for key, module in
                                         policy.parameter_groups().items() if key != "gain_head"},
            "nominal_policy_has_no_gradients": all(p.grad is None for name, module in policy.parameter_groups().items()
                                                   if name != "gain_head" for p in module.parameters()),
            "world_model_unchanged": initial_model == state_digest(model),
            "world_model_has_no_gradients": all(p.grad is None for p in model.parameters()),
            "gain_head_changed": initial_head != state_digest(head),
            "normalizers_unchanged": normalizers == {key: state_digest(module) for key, module in (
                ("observation", policy.tracker.obs_norm), ("goal", policy.tracker.goal_norm))},
        }
        dump_json(output / "checks.json", checks)
        if not all(checks.values()):
            raise RuntimeError("Gain-only pilot contract failed; inspect checks.json")
        save_policy(output / "policy.pt", policy, config=config, training=metadata("complete"))
        dump_json(output / "STATUS.json", metadata("complete"))
        tracker.finish("complete", checks=checks)
    except BaseException as error:
        dump_json(output / "STATUS.json", {**metadata("incomplete"), "error": repr(error)})
        if tracker is not None:
            tracker.fail(error)
        raise


if __name__ == "__main__":
    main()
