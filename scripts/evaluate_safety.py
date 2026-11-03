"""Evaluate the nominal neural policy on identical throws; no QP or world model."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", help="Full-policy checkpoint; omit to evaluate pretrained SuperMimic")
parser.add_argument("--pretrained", default="checkpoints/g1-lafan1-040000")
parser.add_argument("--label")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--limit", type=int, default=0, help="Mark a deliberately partial manifest smoke test")
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np
import torch
from supermimic.env import make_sim

from safety_mimic.evaluation import (
    EpisodeEvidence,
    dump_json,
    rate_summary,
    select_clips,
    sha256,
    validate_manifest,
)
from safety_mimic.experiment import Controller, resolve_artifact
from safety_mimic.isaac_env import DodgeballSim
from safety_mimic.provenance import write_provenance
from safety_mimic.runtime import configure_runtime

ROOT = Path(__file__).resolve().parents[1]


def numpy(value):
    return value.detach().cpu().numpy()


def main():
    configure_runtime(args.device)
    if args.num_envs < 1 or args.limit < 0:
        raise ValueError("Invalid environment count or throw limit")
    start = time.monotonic()
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    throws = manifest["throws"][: args.limit] if args.limit else manifest["throws"]
    args.output = resolve_artifact(ROOT, args.output).resolve()
    if not args.output.is_relative_to(ROOT / "outputs/runs"):
        raise ValueError("Evaluation output must be inside outputs/runs")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a fresh evaluation directory")
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output / "trajectories"
    trajectory_dir.mkdir()
    controller = Controller(ROOT, checkpoint=args.checkpoint, pretrained=args.pretrained, seed=args.seed)
    # An executable deployment contract, including accidental future imports.
    if "qpth.qp" in sys.modules or "safety_mimic.layer" in sys.modules:
        raise RuntimeError("Policy-only evaluation unexpectedly imported the QP layer")
    config = {
        "format": "safetymimic.evaluation.v2",
        "controller": controller.metadata,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "label": args.label or args.output.name,
        "split": manifest["split"],
        "sampling": manifest["sampling"],
        "num_envs": args.num_envs,
        "requested_throws": len(throws),
        "manifest_throws": len(manifest["throws"]),
        "limit": args.limit,
        "episode_seconds": manifest["episode_seconds"],
    }
    dump_json(args.output / "config.json", config)
    sources = (
        [controller.checkpoint]
        if controller.checkpoint
        else [resolve_artifact(ROOT, args.pretrained) / "policy.safetensors"]
    )
    write_provenance(args.output, ROOT, sources)
    sim_cfg = make_sim("cuda:0", args)
    sim_cfg.dt, sim_cfg.physics.num_substeps = 0.005, 1
    episodes, forward_times = [], []
    transitions = 0
    torch.cuda.reset_peak_memory_stats()

    def persist(status):
        dump_json(args.output / "episodes.json", episodes)
        dump_json(
            args.output / "summary.json",
            {
                "label": config["label"],
                "split": manifest["split"],
                "status": status,
                "manifest_coverage_complete": len(episodes) == len(manifest["throws"]),
                "requested_coverage_complete": len(episodes) == len(throws),
                "full_physical_observations": bool(episodes)
                and all(e["physical_outcome_observed"] for e in episodes),
                "rates": rate_summary(episodes, random_sample=manifest["sampling"] == "stratified_random"),
                "simulator_transitions": transitions,
                "physics_substeps": transitions * 4,
                "wall_seconds": time.monotonic() - start,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "actor_batch_seconds_p50": float(np.median(forward_times)) if forward_times else None,
                "deployment": {"uses_qp": False, "uses_world_model": False},
            },
        )
        dump_json(args.output / "selected_clips.json", select_clips(episodes))

    try:
        with launch_simulation(sim_cfg, args):
            env = DodgeballSim(
                sim_cfg,
                args.num_envs,
                controller.motion,
                controller.mdp,
                controller.geometry,
                seed=10000,
                diagnostics=True,
            )
            geometry = controller.geometry
            links = [geometry.model.body(int(geometry.model.geom_bodyid[g])).name for g in geometry.geom_ids]
            dump_json(
                args.output / "geometry.json",
                {
                    "links": links,
                    "names": geometry.geom_names,
                    "radii": numpy(geometry.radii).tolist(),
                    "contact_links": env.contact_sensor.filter_object_names,
                },
            )
            for offset in range(0, len(throws), args.num_envs):
                batch = throws[offset : offset + args.num_envs]
                padded = batch + [batch[-1]] * (args.num_envs - len(batch))
                env.load_throws(padded, manifest["initial"])
                history = None
                evidence = [EpisodeEvidence(row) for row in batch]
                traces, rows, events = [{} for _ in batch], [[] for _ in batch], [[] for _ in batch]

                batch_size = len(batch)

                def append_pose(batch_size=batch_size, saved_traces=traces):
                    snapshot = env.snapshot()
                    state, ball = env.state(), env.ball_state()
                    centers, _ = geometry.centers_and_velocities(state)
                    clearance, _ = geometry.barriers(state, ball)
                    snapshot.update(ball_v=ball.velocity, geom_centers=centers, clearances=clearance)
                    for key, value in snapshot.items():
                        array = numpy(value)
                        for index in range(batch_size):
                            saved_traces[index].setdefault(key, []).append(array[index].copy())

                append_pose()
                aborted = False
                for step in range(round(manifest["episode_seconds"] / 0.02)):
                    with torch.no_grad():
                        state, ball = env.state(), env.ball_state()
                        before, _ = geometry.barriers(state, ball)
                        torch.cuda.synchronize()
                        tick = time.monotonic()
                        try:
                            nominal, history = controller.nominal(env, history)
                            action = controller.execute(env, nominal)
                        except (RuntimeError, ValueError) as error:
                            for item in evidence:
                                item.policy_error = str(error)[:1500]
                            aborted = True
                            break
                        torch.cuda.synchronize()
                        elapsed = time.monotonic() - tick
                        forward_times.append(elapsed)
                        if not bool(torch.isfinite(nominal).all() & torch.isfinite(action).all()):
                            invalid = ~torch.isfinite(nominal).all(-1) | ~torch.isfinite(action).all(-1)
                            for index, item in enumerate(evidence):
                                if bool(invalid[index]):
                                    item.invalid = True
                                    item.event("invalid_action", step * 0.02)
                                else:
                                    item.policy_error = "Batch stopped because another actor action was nonfinite"
                            aborted = True
                            break
                        transition = env.step(action)
                        transitions += args.num_envs
                        actual_state, actual_ball = env.state(), env.ball_state()
                        after, _ = geometry.barriers(actual_state, actual_ball)
                        lower, upper = controller.bounds(env)
                        for index, item in enumerate(evidence):
                            contact = transition["contact_events"][index]
                            events[index].extend(contact)
                            row = {
                                "throw_id": item.throw["throw_id"],
                                "step": step,
                                "time": step * 0.02,
                                "time_end": (step + 1) * 0.02,
                                "clearance_before_m": float(before[index].min()),
                                "clearance_after_m": float(after[index].min()),
                                "clearance_min_m": min(
                                    float(before[index].min()),
                                    float(transition["substep_min_clearance"][index]),
                                ),
                                "threatened_link": links[int(before[index].argmin())],
                                "contact": bool(contact),
                                "fall": bool(transition["fell"][index]),
                                "action_norm": float(action[index].norm()),
                                "actuator_clipping_norm": float((action[index] - nominal[index]).norm()),
                                "action_saturation_count": int(
                                    (
                                        (action[index] <= lower[index] + 1e-4)
                                        | (action[index] >= upper[index] - 1e-4)
                                    ).sum()
                                ),
                                "actor_batch_seconds": elapsed,
                                "root_height_m": float(actual_state[0][index, 2]),
                                "root_tilt_cos": float(actual_state[1][index, 2, 2]),
                            }
                            item.observe(row, contact_events=contact)
                            rows[index].append(row)
                            for key, value in (("nominal", nominal), ("action", action)):
                                traces[index].setdefault(key, []).append(numpy(value[index]).copy())
                        append_pose()
                for index, item in enumerate(evidence):
                    throw_id = item.throw["throw_id"]
                    relative = f"trajectories/{throw_id}.npz"
                    values = {key: np.asarray(value) for key, value in traces[index].items()}
                    values["time"] = np.arange(len(values["root_p"])) * 0.02
                    np.savez_compressed(args.output / relative, **values)
                    if rows[index]:
                        with (trajectory_dir / f"{throw_id}.csv").open("w", newline="") as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(rows[index][0]))
                            writer.writeheader()
                            writer.writerows(rows[index])
                    dump_json(trajectory_dir / f"{throw_id}_contacts.json", events[index])
                    initial_xy = torch.tensor(manifest["initial"]["state"][0][:2], device="cuda:0")
                    displacement = float((env.state()[0][index, :2] - initial_xy).norm())
                    episodes.append(
                        item.finish(
                            duration=len(rows[index]) * 0.02,
                            root_displacement=displacement,
                            completed=not aborted,
                            trajectory=relative,
                        )
                    )
                persist("running")
                print(
                    json.dumps(
                        {
                            "label": config["label"],
                            "finished": len(episodes),
                            "requested": len(throws),
                            "successes": sum(e["physical_task_success"] for e in episodes),
                        }
                    ),
                    flush=True,
                )
        expected_ids = [row["throw_id"] for row in throws]
        if [episode["throw_id"] for episode in episodes] != expected_ids:
            raise RuntimeError("Evaluation throw coverage differs from the manifest")
        persist("complete")
        dump_json(
            args.output / "COMPLETE.json",
            {
                "requested_throws": len(throws),
                "manifest_sha256": config["manifest_sha256"],
                "full_manifest": len(throws) == len(manifest["throws"]),
                "unobserved_outcomes": sum(not e["physical_outcome_observed"] for e in episodes),
                "deployment_uses_qp": False,
                "deployment_uses_world_model": False,
            },
        )
    except BaseException as error:
        persist("incomplete")
        dump_json(args.output / "INCOMPLETE.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
