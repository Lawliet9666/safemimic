import copy

import pytest

from safety_mimic.pilot_gate import assess_pilot


def inputs():
    rows = [{"throw_id": str(i), "contact": i < 7, "fall": False, "physical_task_success": i == 7}
            for i in range(8)]
    before = {"episodes": rows, "config": {"manifest_sha256": "same"},
              "summary": {"status": "complete", "requested_coverage_complete": True,
                          "full_physical_observations": True, "deployment": {"uses_qp": True},
                          "filter_failures": 100, "filter_actions": 1200}}
    after = copy.deepcopy(before)
    after["episodes"][6].update(contact=False, physical_task_success=True)
    after["summary"]["filter_failures"] = 90
    standing = {"passed": True, "steps": 300, "num_envs": 8, "tracking_failures": [False] * 8,
                "mean_normalized_tracking_return": .8}
    training = {"status": {"status": "complete", "updates": 100,
                           "objective": "safetymimic.filter_ball_avoidance.v1", "collection_uses_qp": True,
                           "deployment_uses_qp": True, "imagined_uses_qp": False, "qp_training_gradient": False},
                "checks": {"world_model_unchanged": True, "world_model_has_no_gradients": True,
                           "policy_normalizers_unchanged": True, "critic_parameters_changed": True,
                           "parameter_groups_changed": {"encoder": True, "decoder": True, "ball": True}},
                "metrics": [{"baseline_avoidance": .1, "loss": 1., "gradient_norm": 2., "critic_loss": .2}] * 100}
    return before, after, standing, copy.deepcopy(standing), training


def test_gate_accepts_a_complete_improved_pilot():
    result = assess_pilot(*inputs())
    assert result["passed"] and result["decision"] == "continue_toward_3000"


@pytest.mark.parametrize("failure", ["standing", "filter", "falls", "coverage", "throws", "objective", "avoidance"])
def test_gate_holds_when_a_required_condition_fails(failure):
    before, after, standing_before, standing_after, training = inputs()
    if failure == "standing":
        standing_after["mean_normalized_tracking_return"] = .79
    elif failure == "filter":
        after["summary"]["filter_failures"] = 101
    elif failure == "falls":
        after["episodes"][0]["fall"] = True
    elif failure == "coverage":
        after["summary"]["full_physical_observations"] = False
    elif failure == "throws":
        after["episodes"][0]["throw_id"] = "other"
    elif failure == "objective":
        training["status"]["objective"] = "old_tracking_only"
    elif failure == "avoidance":
        training["metrics"] = [{"baseline_avoidance": 0., "loss": 1., "gradient_norm": 2., "critic_loss": .2}] * 100
    result = assess_pilot(before, after, standing_before, standing_after, training)
    assert not result["passed"] and result["decision"] == "hold"
