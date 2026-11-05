"""Optional W&B curves with explicit identity, run identity and failure status."""

import importlib
import json
import math
import os
import uuid
from pathlib import Path


def add_tracking_arguments(parser):
    parser.add_argument("--wandb_mode", choices=["disabled", "online", "offline"],
                        default=os.environ.get("WANDB_MODE", "disabled"))
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "safetymimic"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_name")
    parser.add_argument("--wandb_expected_email", default=os.environ.get("WANDB_EXPECTED_EMAIL"),
                        help="Verify the authenticated account before uploading; never an API key")


def account_identity(expected_email=None):
    wandb = importlib.import_module("wandb")
    host = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    if not wandb.login(host=host, verify=True, prompt=False):
        raise RuntimeError("No W&B login is available; complete wandb login directly in the terminal")
    api = wandb.Api(overrides={"base_url": host}, timeout=15)
    viewer = api.viewer
    def field(name):
        return viewer.get(name) if isinstance(viewer, dict) else getattr(viewer, name)
    identity = {"email": field("email"), "username": field("username"), "entity": api.default_entity}
    if expected_email and (identity["email"] or "").strip().casefold() != expected_email.strip().casefold():
        raise ValueError("The authenticated W&B account does not match the requested email; relogin in the terminal")
    return identity


def scalar_metrics(prefix, values):
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            result.update(scalar_metrics(name, value))
        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite tracking metric: {name}")
            result[name] = value
    return result


class RunTracker:
    def __init__(self, output, config, *, mode="disabled", project="safetymimic", entity=None,
                 name=None, expected_email=None, resume=False):
        self.run = None
        self.output = Path(output)
        self.path = self.output / "wandb_run.json"
        if mode == "disabled":
            return
        if mode not in {"online", "offline"}:
            raise ValueError("Unsupported W&B mode")
        if resume and mode == "offline":
            raise ValueError("Offline W&B run resume is unsupported; sync the run and use online mode")
        previous = json.loads(self.path.read_text()) if self.path.exists() else None
        if resume and previous is None:
            raise ValueError("No saved W&B run identity; import existing JSON metrics before resuming tracking")
        if previous and not resume:
            raise ValueError("W&B identity already exists; resume it instead of duplicating the run")
        if previous and (previous["project"] != project or (entity and previous["entity"] != entity)):
            raise ValueError("W&B resume entity/project differs from the saved run")
        entity = entity or (previous["entity"] if previous else None)
        if mode == "online":
            identity = account_identity(expected_email)
            entity = entity or (previous["entity"] if previous else identity["entity"])
        wandb = importlib.import_module("wandb")
        run_id = previous["id"] if previous else uuid.uuid4().hex[:8]
        clean_config = {key: value for key, value in config.items()
                        if key != "wandb_expected_email" and not any(s in key.lower() for s in ("api_key", "password", "token"))}
        self.output.mkdir(parents=True, exist_ok=True)
        run_name = name or (previous.get("name") if previous else None) or self.output.name
        self.run = wandb.init(project=project, entity=entity, name=run_name,
                              id=run_id, resume=("must" if resume else "never") if mode == "online" else None, mode=mode,
                              config=clean_config, dir=str(self.output), save_code=False,
                              allow_val_change=resume,
                              settings=wandb.Settings(console="off", init_timeout=60,
                                  base_url=os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")))
        self.run.define_metric("update")
        self.run.define_metric("train/*", step_metric="update")
        self.run.define_metric("rollout/*", step_metric="update")
        record = {"id": self.run.id, "project": self.run.project, "entity": self.run.entity,
                  "name": self.run.name, "url": self.run.url if mode == "online" else None, "mode": mode}
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(self.path)
        print(f"W&B tracking: {record['url'] or 'offline: ' + str(self.output)}", flush=True)

    @classmethod
    def from_args(cls, output, config, args, *, resume=False):
        return cls(output, config, mode=args.wandb_mode, project=args.wandb_project,
                   entity=args.wandb_entity, name=args.wandb_name,
                   expected_email=args.wandb_expected_email, resume=resume)

    def log_training(self, row):
        if self.run is not None:
            self.run.log({"update": row["updates"], **scalar_metrics("train", row)})

    def log_rollout(self, row, update):
        if self.run is None:
            return
        values = scalar_metrics("rollout", row)
        count = len(row["throw_ids"])
        for key in ("contacts", "falls", "successful_dodges"):
            values[f"rollout/{key}_rate"] = row[key] / count
        values["rollout/filter_failure_rate"] = row["filter_failures"] / row["filter_actions"]
        self.run.log({"update": update, **values})

    def log_evaluation(self, phase, evaluation, standing=None):
        if self.run is None:
            return
        values = scalar_metrics(f"eval/{phase}", evaluation)
        if standing is not None:
            values.update(scalar_metrics(f"standing/{phase}", standing))
        # Before/after evaluation is a comparison, not another training update.
        self.run.summary.update(values)

    def log_assessment(self, assessment):
        if self.run is not None:
            self.run.summary.update({"assessment/decision": assessment["decision"],
                                     "assessment/passed": assessment["passed"],
                                     **scalar_metrics("assessment/criteria", assessment["criteria"])})

    def finish(self, status, *, checks=None, error=None):
        if self.run is None:
            return
        values = {"training/status": status, "training/completed": status == "complete"}
        if checks:
            values.update(scalar_metrics("checks", checks))
        if error:
            values["training/error"] = str(error)
        self.run.summary.update(values)
        self.run.finish(exit_code=0 if status in {"complete", "stopped"} else 1)
        self.run = None

    def fail(self, error):
        try:
            self.finish("incomplete", error=repr(error))
        except Exception as logging_error:
            error.add_note(f"W&B finalization also failed: {logging_error!r}")
