"""Upload saved training curves and evaluations without running simulation."""

import argparse
import json
from pathlib import Path

from safety_mimic.tracking import RunTracker, account_identity, add_tracking_arguments

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--check_account", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append evaluation without replaying training history")
    parser.add_argument("--assessment", type=Path, help="Include the physical performance gate independently of completion")
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if args.check_account:
        print(json.dumps(account_identity(args.wandb_expected_email)), flush=True)
        return
    if args.run is None or args.wandb_mode == "disabled":
        raise ValueError("Provide --run and --wandb_mode online or offline")
    root = args.run if args.run.is_absolute() else ROOT / args.run
    config = json.loads((root / "config.json").read_text())
    status = json.loads((root / "STATUS.json").read_text())
    if status["status"] == "running":
        raise ValueError("Do not backfill an active training run")
    config = {**config, "recording_mode": "evaluation_append" if args.resume else "historical_import",
              "original_training_status": status["status"]}
    provenance = sorted(root.glob("provenance*.json"))
    if provenance:
        config["original_provenance"] = json.loads(provenance[0].read_text())
    tracker = RunTracker.from_args(root, config, args, resume=args.resume)
    try:
        if not args.resume:
            metrics = [json.loads(row) for row in (root / "metrics.jsonl").read_text().splitlines()]
            rollout_path = root / "rollouts.jsonl"
            rollouts = ({row["round"]: row for row in (json.loads(line) for line in rollout_path.read_text().splitlines())}
                        if rollout_path.exists() else {})
            recorded_rounds = set()
            for row in metrics:
                round_id = row.get("round")
                if round_id in rollouts and round_id not in recorded_rounds:
                    tracker.log_rollout(rollouts[round_id], row["updates"] - 1)
                    recorded_rounds.add(round_id)
                tracker.log_training(row)
        for phase in ("before", "after"):
            evaluation = root / f"eval_{phase}/summary.json"
            standing = root / f"standing_{phase}.json"
            if evaluation.exists():
                tracker.log_evaluation(phase, json.loads(evaluation.read_text()),
                                       json.loads(standing.read_text()) if standing.exists() else None)
        checks = root / "checks.json"
        if args.assessment:
            tracker.log_assessment(json.loads(args.assessment.read_text()))
        tracker.finish(status["status"], checks=json.loads(checks.read_text()) if checks.exists() else None,
                       error=status.get("error"))
    except BaseException as error:
        tracker.fail(error)
        raise


if __name__ == "__main__":
    main()
