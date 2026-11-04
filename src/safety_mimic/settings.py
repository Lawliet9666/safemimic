"""Shared command-line defaults for training and its numerical validation."""

WORLD_MODEL_PATH = "outputs/checkpoints/world_model_broad.pt"
SAFETY_DEFAULTS = {
    "linearization_steps": 16,
    "step_trust_radius": 0.15,
    "backtracking_steps": 8,
    "velocity_time": 0.25,
    "anticipation_mode": "threat",
    "alpha": 0.2,
    "residual_margin": 0.002,
    "trust_radius": 1.0,
}


def add_safety_arguments(parser):
    for name, default in SAFETY_DEFAULTS.items():
        options = {"choices": ["all", "threat"]} if name == "anticipation_mode" else {}
        parser.add_argument(f"--{name}", type=type(default), default=default, **options)
