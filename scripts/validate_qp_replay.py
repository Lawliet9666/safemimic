"""Replay saved QP inputs to compare old local linearization with the new solve."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from supermimic.networks import WorldModel
from supermimic.robot import G1_JOINT_NAMES

from safety_mimic.ball import BallState
from safety_mimic.geometry import CollisionGeometry
from safety_mimic.layer import SafetyConfig, SafetyLayer
from safety_mimic.runtime import configure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--world_model", type=Path, default=Path("outputs/checkpoints/world_model_broad.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require_feasible", action="store_true", help="Also reject any remaining failed action")
    args = parser.parse_args()
    configure_runtime()
    records = torch.load(args.inputs, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.world_model, map_location="cuda:0", weights_only=True)
    model = WorldModel(hidden=checkpoint["hidden"]).cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["wm"])
    geometry = CollisionGeometry("vendor/supermimic/data/g1_description/g1_29dof.xml",
                                 G1_JOINT_NAMES, device="cuda:0")
    rows = []
    for index, record in enumerate(records):
        state = tuple(value.cuda() for value in record["state"])
        ball = BallState(**{key: value.cuda() for key, value in record["ball"].items()})
        nominal, context, lower, upper = [record[key].cuda() for key in ("nominal", "context", "lower", "upper")]
        config = SafetyConfig(**{**record["config"], "linearization_steps": 16,
                                 "step_trust_radius": 0.15, "backtracking_steps": 8})
        nominal = nominal.requires_grad_(True)
        result = SafetyLayer(model, geometry, config)(state, ball, nominal, context, lower, upper)
        weights = torch.linspace(-1, 1, nominal.shape[-1], device=nominal.device)[None]
        gradient = torch.autograd.grad((result.qp.action * weights).sum(), nominal)[0]
        # Compare both training variants with the same graph construction.
        surrogate = SafetyLayer(model, geometry, config)(state, ball, nominal, context, lower, upper,
                                                        backward="identity")
        row = {
            "case": index, "before_failed": record["failed"].tolist(), "after_failed": result.failed.tolist(),
            "before_nonlinear": record["nonlinear"].amin(-1).tolist(),
            "after_nonlinear": result.nonlinear_residual.amin(-1).detach().tolist(),
            "after_affine": result.qp.affine_residual.amin(-1).detach().tolist(),
            "after_slack": result.qp.slack.amax(-1).detach().tolist(),
            "iterations": result.iterations.tolist(), "gradient_finite": bool(torch.isfinite(gradient).all()),
            "gradient_norm": float(gradient.norm()),
            "identity_forward_max_difference": float((result.qp.action - surrogate.qp.action).detach().abs().max()),
            "failure_meaning": "no feasible action found within iteration budget" if bool(result.failed.any()) else None,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    all_valid = not any(any(row["after_failed"]) for row in rows)
    report = {"config": asdict(config), "episodes": rows, "all_actions_valid": all_valid,
              "scope": "numerical regression; not physical success"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(row["gradient_finite"] and row["identity_forward_max_difference"] < 1e-5 for row in rows):
        raise AssertionError("QP gradient/forward regression failed")
    if args.require_feasible and not all_valid:
        raise AssertionError("QP regression still contains failed actions")


if __name__ == "__main__":
    main()
