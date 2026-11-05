import json
import sys
from types import SimpleNamespace

import pytest

from safety_mimic.tracking import RunTracker, account_identity, scalar_metrics


@pytest.fixture
def sdk(monkeypatch):
    runs, calls = [], []

    class Run:
        def __init__(self, arguments):
            self.id = arguments["id"]
            self.project = arguments["project"]
            self.entity = arguments["entity"] or "researcher"
            self.name = arguments["name"]
            self.url = f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.id}"
            self.summary, self.logs, self.metrics, self.exits = {}, [], [], []

        def define_metric(self, *args, **kwargs):
            self.metrics.append((args, kwargs))

        def log(self, values):
            self.logs.append(values)

        def finish(self, exit_code):
            self.exits.append(exit_code)

    def init(**kwargs):
        calls.append(kwargs)
        run = Run(kwargs)
        runs.append(run)
        return run

    module = SimpleNamespace(init=init, login=lambda **kwargs: True, Settings=lambda **kwargs: kwargs,
                             Api=lambda **kwargs: SimpleNamespace(
                                 viewer=SimpleNamespace(email="owner@example.com", username="researcher"),
                                 default_entity="researcher"))
    monkeypatch.setitem(sys.modules, "wandb", module)
    return SimpleNamespace(runs=runs, calls=calls, module=module)


def test_disabled_tracking_never_imports_sdk_or_creates_files(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Disabled tracking imported a dependency")
    monkeypatch.setattr("safety_mimic.tracking.importlib.import_module", forbidden)
    tracker = RunTracker(tmp_path / "absent", {}, mode="disabled")
    tracker.log_training({"updates": 1})
    tracker.finish("complete")
    assert not (tmp_path / "absent").exists()


def test_requested_account_mismatch_blocks_upload(tmp_path, sdk):
    with pytest.raises(ValueError, match="requested email"):
        RunTracker(tmp_path, {}, mode="online", expected_email="different@example.com")
    assert sdk.calls == []
    assert account_identity("OWNER@example.com")["username"] == "researcher"


def test_training_axes_rollout_rates_and_failure_state_remain_distinct(tmp_path, sdk):
    tracker = RunTracker(tmp_path, {"method": "filter", "api_key": "must-not-upload"}, mode="online",
                         expected_email="owner@example.com")
    tracker.log_training({"updates": 25, "loss": .3, "gradient_norm": .2})
    tracker.log_rollout({"round": 1, "throw_ids": ["a", "b"], "contacts": 1, "falls": 0,
                         "successful_dodges": 1, "filter_failures": 3, "filter_actions": 20}, 25)
    tracker.log_assessment({"decision": "hold", "passed": False, "criteria": {"standing": True}})
    tracker.finish("complete", checks={"world_model_unchanged": True})
    run = sdk.runs[0]
    assert run.logs[0] == {"update": 25, "train/updates": 25, "train/loss": .3, "train/gradient_norm": .2}
    assert run.logs[1]["rollout/contacts_rate"] == .5
    assert run.logs[1]["rollout/filter_failure_rate"] == .15
    assert run.summary["training/completed"] and not run.summary["assessment/passed"]
    assert run.summary["assessment/decision"] == "hold" and run.exits == [0]
    assert "api_key" not in sdk.calls[0]["config"]
    assert (("train/*",), {"step_metric": "update"}) in run.metrics


def test_exception_finishes_as_failed_and_online_resume_preserves_identity(tmp_path, sdk):
    tracker = RunTracker(tmp_path, {}, mode="online", entity="researcher")
    record = json.loads((tmp_path / "wandb_run.json").read_text())
    tracker.fail(RuntimeError("simulation failed"))
    assert sdk.runs[0].exits == [1]
    assert sdk.runs[0].summary["training/status"] == "incomplete"
    resumed = RunTracker(tmp_path, {}, mode="online", resume=True)
    assert sdk.calls[1]["id"] == record["id"] and sdk.calls[1]["resume"] == "must"
    assert sdk.calls[1]["entity"] == "researcher"
    resumed.finish("complete")
    with pytest.raises(ValueError, match="entity/project"):
        RunTracker(tmp_path, {}, mode="online", project="other", resume=True)
    with pytest.raises(ValueError, match="Offline"):
        RunTracker(tmp_path, {}, mode="offline", resume=True)


def test_nonfinite_metrics_are_rejected_and_finalization_preserves_original_error(tmp_path, sdk):
    with pytest.raises(ValueError, match="Nonfinite"):
        scalar_metrics("train", {"loss": float("nan")})
    tracker = RunTracker(tmp_path, {}, mode="online")
    def broken(**kwargs):
        raise RuntimeError("sync unavailable")
    sdk.runs[0].finish = broken
    original = RuntimeError("original simulation failure")
    tracker.fail(original)
    assert "sync unavailable" in original.__notes__[0]
