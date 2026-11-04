"""Record the pilot decision; this script never submits a continuation job."""

import argparse
import json
from pathlib import Path

from safety_mimic.pilot_gate import assess_pilot


def read(path):
    return json.loads(path.read_text())


def bundle(path):
    return {name: read(path / f"{name}.json") for name in ("config", "summary", "episodes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--standing_before", type=Path, required=True)
    parser.add_argument("--standing_after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = {"status": read(args.run / "STATUS.json"), "checks": read(args.run / "checks.json"),
                "metrics": [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]}
    inputs = {"before": bundle(args.before), "after": bundle(args.after),
              "standing_before": read(args.standing_before), "standing_after": read(args.standing_after),
              "training": training}
    result = assess_pilot(**inputs)
    result["inputs"] = {key: str(value) for key, value in vars(args).items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rows = ["# Avoidance pilot continuation gate", "", f"Decision: **{result['decision']}**.", "",
            "| Criterion | Passed |", "|---|---|",
            *[f"| {key.replace('_', ' ')} | {value} |" for key, value in result["criteria"].items()], "",
            "| Metric | Before | After 100 updates |", "|---|---:|---:|"]
    for key in ("contact", "fall", "physical_task_success"):
        rows.append(f"| {key} | {result['counts']['before'][key]} | {result['counts']['after'][key]} |")
    rows.extend([f"| failed filter actions | {result['filter_failures'][0]} | {result['filter_failures'][1]} |",
                 f"| standing normalized return | {result['standing_returns'][0]:.8f} | {result['standing_returns'][1]:.8f} |",
                 "", "Standing return tolerance: 0.0001 absolute (0.01 percentage points). "
                 "Standing uses the unfiltered actor for 300 steps in eight ball-free environments.", "",
                 "Both throw evaluations use the same first eight development throws and the same execution filter. "
                 "This is a continuation decision for simulation training, not evidence of real-time deployment safety.", ""])
    args.output.with_suffix(".md").write_text("\n".join(rows))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
