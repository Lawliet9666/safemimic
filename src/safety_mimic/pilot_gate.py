"""Conservative continuation decision from paired physical pilot observations."""

import math

STANDING_RETURN_ATOL = 1e-4


def assess_pilot(before, after, standing_before, standing_after, training):
    first, second = before["episodes"], after["episodes"]
    ids_before = [row["throw_id"] for row in first]
    ids_after = [row["throw_id"] for row in second]
    counts = {name: {key: sum(bool(row[key]) for row in rows)
                     for key in ("contact", "fall", "physical_task_success")}
              for name, rows in (("before", first), ("after", second))}
    summaries = [before["summary"], after["summary"]]
    failures = [row["filter_failures"] for row in summaries]
    actions = [row["filter_actions"] for row in summaries]
    scores = [row["mean_normalized_tracking_return"] for row in (standing_before, standing_after)]
    checks = training["checks"]
    status = training["status"]
    metrics = training["metrics"]
    criteria = {
        "same_eight_fixed_throws": len(first) == len(second) == len(set(ids_before)) == 8
            and ids_before == ids_after
            and before["config"]["manifest_sha256"] == after["config"]["manifest_sha256"],
        "complete_physical_observations": all(row["status"] == "complete"
            and row["requested_coverage_complete"] and row["full_physical_observations"] for row in summaries),
        "external_filter_used_in_both_evaluations": all(row["deployment"]["uses_qp"] for row in summaries)
            and actions[0] == actions[1] and actions[0] > 0,
        "fewer_contacts": counts["after"]["contact"] < counts["before"]["contact"],
        "more_successful_dodges": counts["after"]["physical_task_success"] > counts["before"]["physical_task_success"],
        "no_additional_falls": counts["after"]["fall"] <= counts["before"]["fall"],
        "filter_failures_not_increased": failures[1] <= failures[0],
        "standing_protocol_matched": standing_before["steps"] == standing_after["steps"]
            and standing_before["num_envs"] == standing_after["num_envs"],
        "standing_passed": standing_before["passed"] and standing_after["passed"]
            and not any(standing_after["tracking_failures"]),
        "standing_return_not_degraded": all(math.isfinite(x) for x in scores)
            and scores[1] + STANDING_RETURN_ATOL >= scores[0],
        "pilot_training_completed": status["status"] == "complete" and status["updates"] == 100
            and len(metrics) == 100,
        "correct_objective_and_gradient_route": status["objective"] == "safetymimic.filter_ball_avoidance.v1"
            and status["collection_uses_qp"] and status["deployment_uses_qp"]
            and not status["imagined_uses_qp"] and not status["qp_training_gradient"],
        "avoidance_signal_used": any(row["baseline_avoidance"] > 0 for row in metrics),
        "finite_training": bool(metrics) and all(math.isfinite(row[key]) for row in metrics
            for key in ("loss", "gradient_norm", "critic_loss")),
        "frozen_model_and_normalizers": checks["world_model_unchanged"]
            and checks["world_model_has_no_gradients"] and checks["policy_normalizers_unchanged"],
        "actor_and_critic_updated": all(checks["parameter_groups_changed"].values())
            and checks["critic_parameters_changed"],
    }
    passed = all(criteria.values())
    return {"passed": passed, "decision": "continue_toward_3000" if passed else "hold",
            "criteria": criteria, "counts": counts, "filter_failures": failures, "filter_actions": actions,
            "standing_returns": scores, "standing_return_atol": STANDING_RETURN_ATOL,
            "scope": "eight paired development throws; pilot continuation gate, not a general safety claim"}
