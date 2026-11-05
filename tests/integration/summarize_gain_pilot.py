"""Summarize paired fixed/adaptive-gain physical outcomes; never submit training."""

import argparse
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    config, status, checks = (read(root / f"{key}.json") for key in ("config", "STATUS", "checks"))
    evaluations = [read(root / f"eval_{phase}/summary.json") for phase in ("before", "after")]
    episodes = [read(root / f"eval_{phase}/episodes.json") for phase in ("before", "after")]
    eval_config = [read(root / f"eval_{phase}/config.json") for phase in ("before", "after")]
    standing = [read(root / f"standing_{phase}.json") for phase in ("before", "after")]
    metrics = [json.loads(row) for row in (root / "metrics.jsonl").read_text().splitlines()]
    rollouts = [json.loads(row) for row in (root / "rollouts.jsonl").read_text().splitlines()]
    ids = [[row["throw_id"] for row in batch] for batch in episodes]
    counts = [{key: sum(bool(e[key]) for e in batch) for key in ("contact", "fall", "physical_task_success")}
              for batch in episodes]
    conditions = {
        "matched_eight_throws": ids[0] == ids[1] == config["evaluation_throw_ids"] and len(set(ids[0])) == 8
            and eval_config[0]["manifest_sha256"] == eval_config[1]["manifest_sha256"] == config["manifest_sha256"],
        "evaluation_excluded_from_training": not set(ids[0]) & set(config["training_throw_ids"]),
        "complete_physical_evaluations": all(e["status"] == "complete" and e["full_physical_observations"]
                                             and e["requested_coverage_complete"] for e in evaluations),
        "same_filter_action_count": evaluations[0]["filter_actions"] == evaluations[1]["filter_actions"] > 0
            and all(e["deployment"]["uses_qp"] for e in evaluations),
        "initial_gain_matches_fixed_baseline": abs(evaluations[0]["class_k_gain_mean"] - .2) < 1e-6,
        "fewer_contacts": counts[1]["contact"] < counts[0]["contact"],
        "more_successful_dodges": counts[1]["physical_task_success"] > counts[0]["physical_task_success"],
        "no_additional_falls": counts[1]["fall"] <= counts[0]["fall"],
        "filter_failures_not_increased": evaluations[1]["filter_failures"] <= evaluations[0]["filter_failures"],
        "standing_preserved": all(s["passed"] and not any(s["tracking_failures"]) for s in standing)
            and standing[0]["steps"] == standing[1]["steps"] == 300
            and standing[0]["num_envs"] == standing[1]["num_envs"] == 8
            and standing[1]["mean_normalized_tracking_return"] + 1e-4 >= standing[0]["mean_normalized_tracking_return"],
        "training_completed": status["status"] == "complete" and status["updates"] == len(metrics) == 100
            and len(rollouts) == 4,
        "finite_gain_learning": all(math.isfinite(m[k]) for m in metrics for k in ("loss", "gradient_norm"))
            and any(m["gradient_norm"] > 0 for m in metrics),
        "correct_gradient_route": config["objective"] == "safetymimic.filter_gain_score.v1"
            and not config["qp_training_gradient"] and config["nominal_actor_frozen"],
        "frozen_contract_and_gain_update": all(checks.values()),
    }
    result = {"passed": all(conditions.values()), "decision": "improved_pilot" if all(conditions.values()) else "hold",
              "criteria": conditions, "counts": counts, "evaluations": evaluations, "standing": standing,
              "checks": checks, "training": status, "rollouts": rollouts,
              "median_optimizer_update_seconds": statistics.median(m["update_seconds"] for m in metrics),
              "total_rollout_seconds": sum(r["rollout_seconds"] for r in rollouts),
              "gain_range": config["gain_bounds"], "gain_initial": config["gain_initial"],
              "scope": "eight development throws excluded from gain training; no general safety claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Adaptive class-K filter pilot", "", f"Decision: **{result['decision']}**.", "",
             "The pretrained action actor and world model are frozen. Only the new bounded gain head is trained. "
             "Both evaluations execute the same QP; before training the head outputs the fixed coefficient 0.2. "
             "The 100 updates are PPO gain-head optimizer steps over four fresh, complete simulator rollout batches. "
             "They are not the earlier 100 imagined-world-model actor updates.", "",
             "| Metric | Fixed initial gain | Learned gain |", "|---|---:|---:|"]
    for key in ("contact", "fall", "physical_task_success"):
        lines.append(f"| {key} | {counts[0][key]}/8 | {counts[1][key]}/8 |")
    for key in ("filter_failures", "filter_actions", "class_k_gain_mean", "class_k_gain_min", "class_k_gain_max",
                "controller_batch_seconds_p50", "controller_batch_seconds_p95"):
        lines.append(f"| {key} | {evaluations[0][key]} | {evaluations[1][key]} |")
    lines.append(f"| standing normalized return | {standing[0]['mean_normalized_tracking_return']} | "
                 f"{standing[1]['mean_normalized_tracking_return']} |")
    lines.extend(["", f"Simulator rollout time: {result['total_rollout_seconds']:.3f} s. "
                  f"Median gain optimizer step: {result['median_optimizer_update_seconds']:.6f} s.", "",
                  "| Criterion | Passed |", "|---|---|",
                  *[f"| {name.replace('_', ' ')} | {value} |" for name, value in conditions.items()], "",
                  "QP failures, contacts and falls are independent metrics. Simulated outcomes exclude controller "
                  "wall-clock delays. This report does not submit continuation or establish deployment safety.", ""])
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
