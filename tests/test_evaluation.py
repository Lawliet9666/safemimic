import copy

import numpy as np
import pytest

from safety_mimic.evaluation import (
    REGIONS,
    EpisodeEvidence,
    make_manifest,
    paired_difference,
    rate_summary,
    select_clips,
    validate_manifest,
    wilson,
)


def initial():
    return {
        "state": [[0, 0, 0.8], np.eye(3).tolist(), [0, 0, 0], [0, 0, 0], [0.0] * 29, [0.0] * 29],
        "xi": [1.0] + [0.0] * 32,
        "frame": 0,
        "previous_action": [0.0] * 29,
    }


def test_manifests_balanced_deterministic_and_separate():
    centers = {region: [0, 0, 1] for region in REGIONS}
    dev = validate_manifest(make_manifest("development", initial(), centers))
    val = validate_manifest(make_manifest("validation", initial(), centers))
    assert dev == make_manifest("development", initial(), centers)
    assert len(dev["throws"]) == 96 and len(val["throws"]) == 192
    assert not {t["throw_id"] for t in dev["throws"]} & {t["throw_id"] for t in val["throws"]}
    assert len({(t["region"], t["direction"], t["flight_time_bin"]) for t in dev["throws"]}) == 96
    for row in dev["throws"] + val["throws"]:
        t = row["flight_time"]
        landed = (
            np.array(row["ball_position"])
            + np.array(row["ball_velocity"]) * t
            + np.array([0, 0, -4.905]) * t * t
        )
        np.testing.assert_allclose(landed, row["target"], atol=1e-10)
    malformed = copy.deepcopy(dev)
    malformed["throws"][1]["throw_id"] = malformed["throws"][0]["throw_id"]
    with pytest.raises(ValueError):
        validate_manifest(malformed)


def throw():
    return {"throw_id": "development-000", "region": "head_link", "direction": 0, "flight_time_bin": 0.7}


def row(**overrides):
    return {
        "clearance_min_m": 0.1,
        "clearance_before_m": 0.1,
        "clearance_after_m": 0.1,
        "fall": False,
        "time": 0.0,
        "time_end": 0.02,
        **overrides,
    }


def test_first_contact_and_fall_persist_after_later_recovery():
    e = EpisodeEvidence(throw())
    e.observe(row(fall=True), contact_events=[{"time": 0.005, "link": "head_link"}])
    e.observe(row(time=2.98, time_end=3.0))
    record = e.finish(duration=3, root_displacement=0.01, completed=True, trajectory="a.npz")
    assert not record["physical_task_success"]
    assert record["first_events"]["contact"] == 0.005
    assert record["first_events"]["fall"] == 0.02
    assert record["execution_uses_qp"] is False


def test_invalid_action_is_not_a_contact_free_observation():
    e = EpisodeEvidence(throw())
    e.invalid = True
    record = e.finish(duration=0, root_displacement=0, completed=False)
    summary = rate_summary([record], random_sample=True)
    assert not record["physical_task_success"] and record["recovered"] is None
    assert summary["contact"]["n"] == 0 and summary["contact"]["rate"] is None
    assert summary["physical_task_success"]["rate"] == 0


def test_failed_execution_filter_is_reported_separately_from_physical_contact():
    evidence = EpisodeEvidence(throw(), execution_uses_qp=True)
    evidence.observe(row(qp_failed=True))
    result = evidence.finish(duration=3, root_displacement=0, completed=True)
    assert result["execution_uses_qp"]
    assert result["physical_task_success"] and not result["contact"]
    assert "qp_filter_failure" in result["failure_categories"]


def test_contact_time_is_physics_time_and_conservatism_is_not_contact():
    e = EpisodeEvidence(throw())
    e.observe(
        row(clearance_min_m=-0.02),
        contact_events=[{"time": 0.005, "link": "head_link"}],
    )
    result = e.finish(duration=3, root_displacement=0, completed=True)
    assert result["first_events"]["contact"] == 0.005
    assert result["contact_links"] == ["head_link"]
    assert "proxy_violation_without_contact" not in result["failure_categories"]
    assert not result["execution_uses_qp"]


def test_empty_statistics_and_paired_identity():
    assert wilson(0, 0) is None
    assert wilson(0, 10)[1] > 0
    assert rate_summary([])["physical_task_success"]["rate"] is None
    a = EpisodeEvidence(throw()).finish(duration=3, root_displacement=0, completed=True)
    stats = paired_difference([a], [a])
    assert stats["paired_bootstrap_ci95"] == [0, 0]
    b = {**a, "throw_id": "different"}
    with pytest.raises(ValueError):
        paired_difference([a], [b])
    # Repeated IDs are not additional independent observations.
    with pytest.raises(ValueError):
        paired_difference([a, a], [a, a])


def test_clip_selection_covers_successes_and_failure_categories():
    e = EpisodeEvidence(throw())
    e.observe(row(clearance_min_m=-0.1))
    a = e.finish(duration=3, root_displacement=0, completed=True, trajectory="a.npz")
    e2 = EpisodeEvidence({**throw(), "throw_id": "other"})
    e2.observe(row())
    b = e2.finish(duration=3, root_displacement=0, completed=True, trajectory="b.npz")
    clips = select_clips([a, b])
    reasons = {reason for c in clips for reason in c["reasons"]}
    assert "physical_success" in reasons and "representative:head_link" in reasons


def test_manifest_rejects_shortened_episodes_and_bad_context():
    centers = {region: [0, 0, 1] for region in REGIONS}
    manifest = make_manifest("development", initial(), centers)
    manifest["episode_seconds"] = 0
    with pytest.raises(ValueError, match="three-second"):
        validate_manifest(manifest)
    manifest["episode_seconds"] = 3.0
    manifest["initial"]["xi"] = [1.0]
    with pytest.raises(ValueError, match="context"):
        validate_manifest(manifest)


def test_batch_aborted_peer_remains_unobserved_without_inventing_invalid_action():
    evidence = EpisodeEvidence(throw())
    evidence.policy_error = "Batch aborted because another action was invalid"
    record = evidence.finish(duration=0.02, root_displacement=0, completed=False)
    assert not record["physical_outcome_observed"]
    assert not record["invalid_action"]
    assert not record["physical_task_success"]
    summary = rate_summary([record])
    assert summary["contact"]["n"] == 0
