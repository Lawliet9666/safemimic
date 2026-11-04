"""Record finite-difference convergence and QP branches without changing the filter."""

import argparse
import json
from pathlib import Path

import torch
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import WorldModel
from supermimic.robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list

import safety_mimic.layer as layer_module
from safety_mimic.ball import BallState
from safety_mimic.geometry import CollisionGeometry
from safety_mimic.layer import SafetyConfig, SafetyLayer
from safety_mimic.runtime import configure_runtime
from safety_mimic.settings import SAFETY_DEFAULTS, WORLD_MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["local", "slurm"], default="slurm")
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--world_model", type=Path, default=ROOT / WORLD_MODEL_PATH)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--distances", type=float, nargs="+", default=[3, 1.5, 1, 0.6])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5, 1e-6])
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_runtime(profile=args.runtime, cpu_threads=args.cpu_threads)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    device, dtype = "cuda:0", torch.float64
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(str(ROOT / "data/standing"), kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    geometry = CollisionGeometry(ROOT / "vendor/supermimic/data/g1_description/g1_29dof.xml",
                                 G1_JOINT_NAMES, device=device, dtype=dtype)
    checkpoint = torch.load(args.world_model, map_location=device, weights_only=True)
    model = WorldModel(hidden=checkpoint["hidden"]).to(device=device, dtype=dtype).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["wm"])
    config = {**SAFETY_DEFAULTS, "linearization_steps": args.steps}
    layer = SafetyLayer(model, geometry, SafetyConfig(**config))
    n = len(args.distances)
    state = tuple(x.to(dtype) for x in motion.ref_state(torch.zeros(n, device=device, dtype=torch.long)))
    centers, _ = geometry.centers_and_velocities(state)
    torso = next(i for i, geom in enumerate(geometry.geom_ids)
                 if geometry.model.body(int(geometry.model.geom_bodyid[geom])).name == "torso_link")
    ball = BallState(centers[:, torso] + state[0].new_tensor(args.distances)[:, None] * state[0].new_tensor([1, 0, 0]),
                     state[0].new_tensor([-5, 0, 0]).expand(n, -1), state[0].new_full((n,), 0.08))
    scale = torch.tensor([action_scale(name) for name in G1_JOINT_NAMES], device=device)
    nominal = ((state[4] - mdp.default_q) / scale).detach().requires_grad_(True)
    xi = nominal.new_zeros(n, 33)
    xi[:, 0] = 1
    context = normalize_xi(xi)
    lower = ((geometry.q_lo - mdp.default_q) / scale).clamp_min(-5)
    upper = ((geometry.q_hi - mdp.default_q) / scale).clamp_max(5)
    original = layer_module.project_action
    trace = []

    def indices(mask):
        return [row.nonzero().flatten().tolist() for row in mask]

    def traced(a, jacobian, residual, lo, hi, **options):
        result = original(a, jacobian, residual, lo, hi, **options)
        with torch.no_grad():
            center = a.clamp(lo, hi)
            low, high = torch.maximum(lo, center - options["trust_radius"]), torch.minimum(hi, center + options["trust_radius"])
            if options["step_trust_radius"] is not None:
                anchor = options["linearization"].clamp(lo, hi)
                low = torch.maximum(low, anchor - options["step_trust_radius"])
                high = torch.minimum(high, anchor + options["step_trust_radius"])
            trace.append({"failed": result.failed.tolist(),
                          "active_affine": indices((result.affine_residual + result.slack).abs() < 1e-5),
                          "active_lower": indices((result.action - low).abs() < 1e-5),
                          "active_upper": indices((result.action - high).abs() < 1e-5),
                          "active_slack": indices(result.slack.abs() < 1e-5)})
        return result

    def run(action):
        trace.clear()
        result = layer(state, ball, action, context, lower, upper)
        signature = {"failed": result.failed.tolist(), "iterations": result.iterations.tolist(),
                     "stalled": result.search_stalled.tolist(), "solves": list(trace)}
        return result, signature

    layer_module.project_action = traced
    try:
        result, reference = run(nominal)
        weights = torch.randn_like(nominal)
        gradient = torch.autograd.grad((result.qp.action * weights).sum(), nominal)[0]
        directions = [torch.nn.functional.normalize(torch.randn_like(nominal), dim=-1)
                      for _ in range(args.directions)]
        derivatives = [(gradient * direction).sum(-1).detach() for direction in directions]
        report = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "reference": reference, "gradient_norms": gradient.detach().norm(dim=-1).tolist(),
                  "initial_barrier": result.initial_barrier.amin(-1).detach().tolist(),
                  "initial_residual": result.initial_residual.amin(-1).tolist(),
                  "intervention": (result.qp.action - nominal).detach().norm(dim=-1).tolist(),
                  "analytic": [value.tolist() for value in derivatives], "finite_difference": []}
        del result, gradient
        for index, (direction, analytic) in enumerate(zip(directions, derivatives)):
            for epsilon in args.epsilons:
                values, signatures = [], []
                for sign in (-1, 1):
                    shifted, signature = run(nominal.detach() + sign * epsilon * direction)
                    values.append((shifted.qp.action * weights).sum(-1).detach())
                    signatures.append(signature)
                    del shifted
                numerical = (values[1] - values[0]) / (2 * epsilon)
                row = {"direction": index, "epsilon": epsilon, "numerical": numerical.tolist(),
                       "normalized_error": ((analytic - numerical).abs() / (1 + numerical.abs())).tolist(),
                       "same_signature": [signature == reference for signature in signatures],
                       "signatures": signatures}
                report["finite_difference"].append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "signatures"}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        layer_module.project_action = original


if __name__ == "__main__":
    main()
