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
from safety_mimic.settings import SAFETY_DEFAULTS, WORLD_MODEL_PATH, add_safety_arguments

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["local", "slurm"], default="local")
    parser.add_argument("--cpu_threads", type=int)
    add_safety_arguments(parser)
    parser.add_argument("--world_model", type=Path, default=ROOT / WORLD_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/results/layer_validation_current.json")
    parser.add_argument("--adaptive_gain", action="store_true")
    parser.add_argument("--gain_finite_difference", action="store_true", help="Check the new coefficient derivative only")
    parser.add_argument("--ball_distances", type=float, nargs=4, default=[3., 1.5, 1., .6])
    parser.add_argument("--ball_speed", type=float, default=5.)
    parser.add_argument(
        "--finite_difference",
        action="store_true",
        help="Use float64 and check real-model action sensitivities along random directions",
    )
    args = parser.parse_args()
    if args.gain_finite_difference and not args.adaptive_gain:
        raise ValueError("Gain finite differences require --adaptive_gain")
    configure_runtime(profile=args.runtime, cpu_threads=args.cpu_threads)
    torch.manual_seed(17)
    device = "cuda:0"
    dtype = torch.float64 if args.finite_difference or args.gain_finite_difference else torch.float32
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
        SafetyConfig(**{name: getattr(args, name) for name in SAFETY_DEFAULTS}),
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
    ball_position = centers[:, torso] + torch.tensor(args.ball_distances, device=device)[:, None] * (
        state[0].new_tensor([1, 0, 0])
    )
    ball = BallState(
        ball_position, state[0].new_tensor([-args.ball_speed, 0, 0]).expand(4, -1), state[0].new_full((4,), 0.08)
    )
    # Use the exact upstream actuator mapping.
    from supermimic.robot import action_scale

    scale = torch.tensor([action_scale(name) for name in G1_JOINT_NAMES], device=device)
    nominal = ((state[4] - mdp.default_q) / scale).detach().requires_grad_(True)
    xi = nominal.new_zeros(4, 33)
    xi[:, 0] = 1
    lower = ((geometry.q_lo - mdp.default_q) / scale).clamp_min(-5)
    upper = ((geometry.q_hi - mdp.default_q) / scale).clamp_max(5)
    gain = nominal.new_full((4,), args.alpha, requires_grad=True) if args.adaptive_gain else None
    options = {} if gain is None else {"gain": gain}
    torch.cuda.synchronize()
    start = time.monotonic()
    result = layer(state, ball, nominal, normalize_xi(xi), lower, upper, **options)
    torch.cuda.synchronize()
    forward_seconds = time.monotonic() - start
    weights = torch.randn_like(nominal)
    start = time.monotonic()
    gradients = torch.autograd.grad((result.qp.action * weights).sum(), (nominal,) if gain is None else (nominal, gain))
    gradient = gradients[0]
    torch.cuda.synchronize()
    backward_seconds = time.monotonic() - start
    if not torch.isfinite(gradient).all():
        raise AssertionError("Nonfinite real-model safety-layer gradient")
    surrogate = layer(state, ball, nominal, normalize_xi(xi), lower, upper, backward="identity", **options)
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
        "min_initial_barrier": layer.barrier(state, ball).amin(-1).detach().tolist(),
        "min_initial_geometric_clearance": result.clearance.amin(-1).detach().tolist(),
        "min_nominal_nonlinear_residual": result.initial_residual.amin(-1).tolist(),
        "qp_failures": result.qp.failed.tolist(),
        "min_affine_residual": result.qp.affine_residual.amin(-1).detach().tolist(),
        "max_slack": result.qp.slack.amax(-1).detach().tolist(),
        "min_nonlinear_residual": result.nonlinear_residual.amin(-1).detach().tolist(),
        "intervention_norm": (result.qp.action - nominal).norm(dim=-1).detach().tolist(),
        "scope": "synthetic incoming-ball states; numerical validation, not a success benchmark",
        "failure_meaning": "no unrelaxed feasible action found; not proof of infeasibility",
    }
    report["geometric_constraint"] = True
    report["ball_distances"] = args.ball_distances
    report["ball_speed"] = args.ball_speed
    report["world_model"] = str(args.world_model)
    report["safety_settings"] = {name: getattr(args, name) for name in SAFETY_DEFAULTS}
    if gain is not None:
        report["gain_gradient"] = gradients[1].detach().tolist()
        if not torch.isfinite(gradients[1]).all() or gradients[1].abs().sum() == 0:
            raise AssertionError("Real-model gain derivative is nonfinite or inactive")
        filtered = layer.filter(state, ball, nominal, normalize_xi(xi), lower, upper, gain=gain)
        torch.testing.assert_close(filtered.qp.action, result.qp.action, atol=1e-5, rtol=1e-5)
        if filtered.qp.action.requires_grad:
            raise AssertionError("External filter retained a QP backward graph")
        report["filter_forward_matches_differentiable"] = True
        report["world_model_has_no_gradients"] = all(p.grad is None for p in model.parameters())
    if args.gain_finite_difference:
        epsilon = 1e-4
        outputs = []
        for sign in (-1, 1):
            shifted = layer(state, ball, nominal, normalize_xi(xi), lower, upper,
                            gain=gain.detach() + sign * epsilon)
            outputs.append((shifted.qp.action * weights).sum(-1).detach())
        numerical = (outputs[1] - outputs[0]) / (2 * epsilon)
        error = (gradients[1] - numerical).abs() / (1 + numerical.abs())
        report["gain_finite_difference"] = {"analytic": gradients[1].tolist(), "numerical": numerical.tolist(),
                                             "max_normalized_error": float(error.max()),
                                             "passed": bool((error < .02).all())}
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
                    state, ball, nominal.detach() + sign * epsilon * direction, normalize_xi(xi), lower, upper, **options
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
    if args.gain_finite_difference and not report["gain_finite_difference"]["passed"]:
        raise AssertionError("Real-model gain finite-difference check failed")


if __name__ == "__main__":
    main()
