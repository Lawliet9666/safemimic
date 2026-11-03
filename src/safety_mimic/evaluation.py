"""Reproducible throw protocols, persistent episode evidence, and paired statistics.

This module deliberately has no simulator or GPU dependency. Proxy violations,
physical contacts, and missing observations are different events throughout.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REGIONS = (
    "head_link",
    "torso_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_knee_link",
    "right_knee_link",
)
FLIGHT_TIMES = (0.5, 0.7, 1.0, 1.4)


def dump_json(path, value):
    """Strict JSON, replaced atomically so interrupted reports remain readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_manifest(split, initial, target_centers, *, seed=20261102):
    """A fixed 96-case grid, or two independent random draws in each grid cell."""
    if split not in {"development", "validation"}:
        raise ValueError("Unknown manifest split")
    generator = np.random.default_rng(seed + (1000003 if split == "validation" else 0))
    throws = []
    for region in REGIONS:
        for direction in range(4):
            for flight_time in FLIGHT_TIMES:
                for replicate in range(2 if split == "validation" else 1):
                    angle = direction * math.pi / 2
                    duration = flight_time
                    target = np.array(target_centers[region], dtype=float)
                    if split == "validation":
                        angle += generator.uniform(-math.pi / 18, math.pi / 18)
                        duration *= generator.uniform(0.95, 1.05)
                        target += generator.uniform(-0.02, 0.02, size=3)
                    position = target + [4 * math.cos(angle), 4 * math.sin(angle), 0]
                    velocity = (target - position) / duration + [0, 0, 9.81 * duration / 2]
                    throws.append(
                        {
                            "throw_id": f"{split}-{len(throws):03d}",
                            "region": region,
                            "direction": direction * 90,
                            "azimuth_radians": float(angle),
                            "flight_time_bin": flight_time,
                            "flight_time": float(duration),
                            "distance": 4.0,
                            "replicate": replicate,
                            "target": target.tolist(),
                            "ball_position": position.tolist(),
                            "ball_velocity": velocity.tolist(),
                            "ball_radius": 0.08,
                        }
                    )
    return {
        "version": 1,
        "split": split,
        "sampling": "fixed_grid" if split == "development" else "stratified_random",
        "seed": seed,
        "control_dt": 0.02,
        "physics_dt": 0.005,
        "episode_seconds": 3.0,
        "initial": initial,
        "throws": throws,
    }


def validate_manifest(manifest):
    rows = manifest["throws"]
    ids = [row["throw_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Manifest throw IDs must be nonempty and unique")
    if manifest["control_dt"] != 0.02 or manifest["physics_dt"] != 0.005:
        raise ValueError("Unsupported evaluation timestep")
    if manifest["episode_seconds"] != 3.0:
        raise ValueError("Task success requires three-second episodes")
    initial = manifest["initial"]
    for key, shape in (("xi", (33,)), ("previous_action", (29,))):
        if np.shape(initial[key]) != shape or not np.isfinite(initial[key]).all():
            raise ValueError("Malformed simulator context or previous action")
    shapes = [(3,), (3, 3), (3,), (3,), (29,), (29,)]
    if len(initial["state"]) != len(shapes):
        raise ValueError("Malformed initial robot state")
    for value, shape in zip(initial["state"], shapes):
        if np.shape(value) != shape or not np.isfinite(value).all():
            raise ValueError("Malformed or nonfinite initial robot state")
    for row in rows:
        for name in ("ball_position", "ball_velocity"):
            if np.shape(row[name]) != (3,) or not np.isfinite(row[name]).all():
                raise ValueError("Malformed ball state")
        if row["ball_radius"] != 0.08:
            raise ValueError("The simulator uses an 8 cm ball")
    return manifest


@dataclass
class EpisodeEvidence:
    throw: dict
    first_events: dict = field(default_factory=dict)
    contact_links: set = field(default_factory=set)
    causes: set = field(default_factory=set)
    steps: int = 0
    min_clearance: float | None = None
    invalid: bool = False
    policy_error: str | None = None

    def event(self, name, time):
        self.first_events.setdefault(name, float(time))

    def observe(self, row, *, contact_events=()):
        self.steps += 1
        clearance = row["clearance_min_m"]
        self.min_clearance = clearance if self.min_clearance is None else min(self.min_clearance, clearance)
        for event in contact_events:
            self.event("contact", event["time"])
            self.contact_links.add(event["link"])
        if row["fall"]:
            self.event("fall", row["time_end"])
        if contact_events and row["clearance_before_m"] >= 0 and row["clearance_after_m"] >= 0:
            self.causes.add("intersample_or_geometry_mismatch")

    def finish(self, *, duration, root_displacement, completed, trajectory=None):
        hit = "contact" in self.first_events
        fall = "fall" in self.first_events
        observed = bool(completed and not self.invalid and self.policy_error is None)
        recovered = root_displacement < 0.35 if observed else None
        success = bool(observed and not hit and not fall and recovered)
        if hit:
            self.causes.add("ball_contact")
        if fall:
            self.causes.add("fall")
        if observed and not recovered:
            self.causes.add("recovery_failure")
        if not observed:
            self.causes.add("unobserved_outcome")
        if self.min_clearance is not None and self.min_clearance < 0 and observed and not hit:
            self.causes.add("proxy_violation_without_contact")
        return {
            **{key: self.throw[key] for key in ("throw_id", "region", "direction", "flight_time_bin")},
            "duration_seconds": float(duration),
            "completed": bool(completed),
            "physical_outcome_observed": observed,
            "contact": hit,
            "fall": fall,
            "recovered": recovered,
            "root_displacement_m": root_displacement,
            "physical_task_success": success,
            "execution_uses_qp": False,
            "first_events": self.first_events,
            "contact_links": sorted(self.contact_links),
            "failure_categories": sorted(self.causes),
            "invalid_action": self.invalid,
            "policy_error": self.policy_error,
            "min_proxy_clearance_m": self.min_clearance,
            "trajectory": trajectory,
            "steps": self.steps,
        }


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - half), min(1, center + half)]


def rate_summary(episodes, *, random_sample=False):
    summary = {
        "n": len(episodes),
        "completed": sum(e["completed"] for e in episodes),
        "unobserved_outcomes": sum(not e["physical_outcome_observed"] for e in episodes),
    }
    for key in ("physical_task_success", "contact", "fall"):
        values = [e[key] for e in episodes if e[key] is not None]
        # A censored no-contact episode cannot be counted as an observed miss.
        if key in {"contact", "fall"}:
            values = [e[key] for e in episodes if e["physical_outcome_observed"] or e[key]]
        n, successes = len(values), sum(values)
        summary[key] = {
            "count": successes,
            "n": n,
            "rate": successes / n if n else None,
            "ci95": wilson(successes, n) if random_sample else None,
        }
    return summary


def paired_difference(first, second, key="physical_task_success", *, samples=4000):
    """Stratified paired bootstrap; each resampled observation is an entire throw."""
    a = {row["throw_id"]: row for row in first}
    b = {row["throw_id"]: row for row in second}
    if len(a) != len(first) or len(b) != len(second) or a.keys() != b.keys():
        raise ValueError("Paired comparisons require identical unique throw IDs")
    if any(a[k][key] is None or b[k][key] is None for k in a):
        raise ValueError("Cannot compare unavailable outcomes")
    groups = {}
    for k, row in a.items():
        groups.setdefault((row["region"], row["direction"], row["flight_time_bin"]), []).append(k)
    generator = np.random.default_rng(12345)
    draws = np.zeros(samples)
    for ids in groups.values():
        delta = np.array([float(b[k][key]) - float(a[k][key]) for k in ids])
        draws += delta[generator.integers(len(delta), size=(samples, len(delta)))].sum(1)
    draws /= len(a)
    return {
        "n": len(a),
        "difference_second_minus_first": sum(float(b[k][key]) - float(a[k][key]) for k in a) / len(a),
        "paired_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def select_clips(episodes):
    """Deterministic coverage, including successes and unfavorable cases."""
    chosen = {}

    def add(row, reason):
        if row and row.get("trajectory"):
            chosen.setdefault(row["throw_id"], {"throw_id": row["throw_id"], "reasons": []})[
                "reasons"
            ].append(reason)

    for number, region in enumerate(REGIONS):
        preferred = next(
            (
                e
                for e in episodes
                if e["region"] == region
                and e["direction"] == (number % 4) * 90
                and e["flight_time_bin"] == 0.7
            ),
            None,
        )
        add(
            preferred or next((e for e in episodes if e["region"] == region), None),
            "representative:" + region,
        )
    for category in sorted({c for e in episodes for c in e["failure_categories"]}):
        add(next(e for e in episodes if category in e["failure_categories"]), "failure:" + category)
    timed = [e for e in episodes if "contact" in e["first_events"]]
    if timed:
        add(min(timed, key=lambda e: e["first_events"]["contact"]), "earliest_contact")
    available = [e for e in episodes if e["min_proxy_clearance_m"] is not None]
    if available:
        add(min(available, key=lambda e: e["min_proxy_clearance_m"]), "worst_clearance")
    add(next((e for e in episodes if e["physical_task_success"]), None), "physical_success")
    return list(chosen.values())
