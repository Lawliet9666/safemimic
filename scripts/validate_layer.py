"""Exercise the real G1 learned-model QP, including a complete backward pass."""

import argparse
import json
import time
from pathlib import Path

import torch
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import WorldModel
from supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list

from safety_mimic.ball import BallState
from safety_mimic.geometry import CollisionGeometry
from safety_mimic.layer import SafetyConfig, SafetyLayer
from safety_mimic.runtime import configure_runtime

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linearization_steps", type=int, default=1)
    parser.add_argument("--velocity_time", type=float, default=0.15)
    parser.add_argument("--anticipation_mode", choices=["all", "threat"], default="all")
    parser.add_argument("--residual_margin", type=float, default=0.0)
    parser.add_argument("--world_model", type=Path, default=ROOT / "outputs/checkpoints/world_model.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/results/layer_validation_current.json")
    parser.add_argument(
        "--finite_difference",
        action="store_true",
        help="Use float64 and check real-model action sensitivities along random directions",
    )
    args = parser.parse_args()
    configure_runtime()
    torch.manual_seed(17)
    device = "cuda:0"
    dtype = torch.float64 if args.finite_difference else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(str(ROOT / "data/standing"), kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    geometry = CollisionGeometry(
        ROOT / "vendor/supermimic/data/g1_description/g1_29dof.xml",
        G1_JOINT_NAMES,
        device=device,
        dtype=dtype,
    )
    checkpoint = torch.load(args.world_model, map_location=device, weights_only=True)
    model = (
        WorldModel(hidden=checkpoint["hidden"]).to(device=device, dtype=dtype).eval().requires_grad_(False)
    )
    model.load_state_dict(checkpoint["wm"])
    layer = SafetyLayer(
        model,
        geometry,
        SafetyConfig(
            linearization_steps=args.linearization_steps,
            velocity_time=args.velocity_time,
            anticipation_mode=args.anticipation_mode,
            residual_margin=args.residual_margin,
        ),
    )
    state = tuple(
        value.to(dtype) for value in motion.ref_state(torch.zeros(4, device=device, dtype=torch.long))
    )
    centers, _ = geometry.centers_and_velocities(state)
    torso = next(
        i
        for i, geom in enumerate(geometry.geom_ids)
        if geometry.model.body(int(geometry.model.geom_bodyid[geom])).name == "torso_link"
    )
    ball_position = centers[:, torso] + torch.tensor([3.0, 1.5, 1.0, 0.6], device=device)[:, None] * (
        state[0].new_tensor([1, 0, 0])
    )
    ball = BallState(
        ball_position, state[0].new_tensor([-5, 0, 0]).expand(4, -1), state[0].new_full((4,), 0.08)
    )
    # Use the exact upstream actuator mapping.
    from supermimic.robot import action_scale

    scale = torch.tensor([action_scale(name) for name in G1_JOINT_NAMES], device=device)
    nominal = ((state[4] - mdp.default_q) / scale).detach().requires_grad_(True)
    xi = nominal.new_zeros(4, 33)
    xi[:, 0] = 1
    lower = ((geometry.q_lo - mdp.default_q) / scale).clamp_min(-5)
    upper = ((geometry.q_hi - mdp.default_q) / scale).clamp_max(5)
    torch.cuda.synchronize()
    start = time.monotonic()
    result = layer(state, ball, nominal, normalize_xi(xi), lower, upper)
    torch.cuda.synchronize()
    forward_seconds = time.monotonic() - start
    weights = torch.randn_like(nominal)
    start = time.monotonic()
    gradient = torch.autograd.grad((result.qp.action * weights).sum(), nominal)[0]
    torch.cuda.synchronize()
    backward_seconds = time.monotonic() - start
    if not torch.isfinite(gradient).all():
        raise AssertionError("Nonfinite real-model safety-layer gradient")
    surrogate = layer(state, ball, nominal, normalize_xi(xi), lower, upper, backward="identity")
    forward_difference = float((surrogate.qp.action - result.qp.action).detach().abs().max())
    if forward_difference > 1e-6:
        raise AssertionError("Backward ablation changed the forward safety filter")
    report = {
        "linearization_steps": args.linearization_steps,
        "velocity_time": args.velocity_time,
        "anticipation_mode": args.anticipation_mode,
        "forward_seconds_batch4": forward_seconds,
        "backward_seconds_batch4": backward_seconds,
        "forward_ablation_max_difference": forward_difference,
        "gradient_norm": float(gradient.detach().norm()),
        "finite_gradients": True,
        "filter_failures": result.failed.tolist(),
        "qp_failures": result.qp.failed.tolist(),
        "min_affine_residual": result.qp.affine_residual.amin(-1).detach().tolist(),
        "max_slack": result.qp.slack.amax(-1).detach().tolist(),
        "min_nonlinear_residual": result.nonlinear_residual.amin(-1).detach().tolist(),
        "intervention_norm": (result.qp.action - nominal).norm(dim=-1).detach().tolist(),
        "scope": "synthetic incoming-ball states; numerical validation, not a success benchmark",
    }
    report["geometric_constraint"] = True
    report["world_model"] = str(args.world_model)
    if args.finite_difference:
        errors, estimates = [], []
        epsilon = 1e-4
        for _ in range(3):
            direction = torch.nn.functional.normalize(torch.randn_like(nominal), dim=-1)
            outputs = []
            for sign in (-1, 1):
                # Keep the same derivative construction mode as the reference.
                # Do not compare a graph-enabled forward with a different path.
                shifted = layer(
                    state, ball, nominal.detach() + sign * epsilon * direction, normalize_xi(xi), lower, upper
                )
                outputs.append((shifted.qp.action * weights).sum(-1).detach())
            numerical = (outputs[1] - outputs[0]) / (2 * epsilon)
            analytic = (gradient * direction).sum(-1)
            relative = (analytic - numerical).abs() / (1 + numerical.abs())
            estimates.append({"analytic": analytic.tolist(), "finite_difference": numerical.tolist()})
            errors.extend(relative.tolist())
        report["finite_difference"] = {
            "epsilon": epsilon,
            "dtype": "float64",
            "max_normalized_error": max(errors),
            "directions": estimates,
            "passed": max(errors) < 0.02,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.finite_difference and not report["finite_difference"]["passed"]:
        raise AssertionError(
            "Real-model finite-difference check failed; inspect active-set degeneracy and conditioning"
        )


if __name__ == "__main__":
    main()
