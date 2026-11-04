"""Summarize measured matched settings; keep published and historical references separate."""

import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = {"root": str(args.root), "source_commit": (args.root / "source_commit.txt").read_text().strip(),
              "scope": "short end-to-end pilot; replay diverges with filtered execution; not a convergence benchmark",
              "modes": {}}
    settings, initial = [], []
    for mode in ("supermimic", "filter", "safetymimic"):
        root = args.root / mode
        status = read(root / "STATUS.json")
        if status["status"] != "complete":
            raise RuntimeError(f"Incomplete comparison mode: {mode}")
        config, checks = read(root / "config.json"), read(root / "checks.json")
        if not checks["world_model_unchanged"] or not checks["world_model_has_no_gradients"]:
            raise RuntimeError(f"Frozen-model contract failed: {mode}")
        if not checks["policy_normalizers_unchanged"] or not all(checks["parameter_groups_changed"].values()):
            raise RuntimeError(f"Actor/normalizer checks failed: {mode}")
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        settings.append({key: config[key] for key in ("world_size", "global_batch_size", "num_envs", "horizon",
                        "gradient_accumulation_steps", "warmup_steps", "seed", "actor_lr", "ball_lr", "updates")})
        initial.append((status["world_model_sha256_initial"], status["initial_parameter_group_sha256"]))
        record["modes"][mode] = {"settings": settings[-1], "updates": rows, "checks": checks,
            "warmup": read(root / "warmup.json"), "evaluation": read(args.root / f"eval_{mode}/summary.json"),
            "median_seconds": {key: statistics.median(row[key] for row in rows) for key in
                               ("update_seconds", "collection_seconds", "collection_filter_seconds",
                                "forward_seconds", "backward_seconds", "critic_seconds")},
            "peak_gpu_allocated_gib": max(row["gpu_peak_allocated_bytes"] for row in rows) / 1024**3}
    if any(value != settings[0] for value in settings[1:]) or any(value != initial[0] for value in initial[1:]):
        raise RuntimeError("Hardware configuration, initial actor, or world model differs between modes")
    record["single_environment_filter_evaluation"] = read(args.root / "eval_filter_batch1/summary.json")
    baseline = record["modes"]["supermimic"]["median_seconds"]["update_seconds"]
    columns = ["supermimic", "filter", "safetymimic"]
    def values(key):
        return [record["modes"][mode]["median_seconds"][key] for mode in columns]
    table = [(key.replace("_", " "), [f"{value:.3f}" for value in values(key)]) for key in
             ("collection_seconds", "collection_filter_seconds", "forward_seconds", "backward_seconds",
              "critic_seconds", "update_seconds")]
    table.extend([
        ("updates per hour", [f"{3600 / value:.2f}" for value in values("update_seconds")]),
        ("time ratio to measured no-QP baseline", [f"{value / baseline:.2f}x" for value in values("update_seconds")]),
        ("peak GPU allocation (GiB)", [f"{record['modes'][mode]['peak_gpu_allocated_gib']:.2f}" for mode in columns]),
        ("warmup seconds", [f"{record['modes'][mode]['warmup']['collection_seconds']:.3f}" for mode in columns]),
    ])
    text = ["# Safety-filter timing comparison", "", f"Source: `{record['source_commit']}`.", "",
            "All training modes use two GPUs, horizon 32, global actor batch 32, 64 simulator environments per GPU, "
            "48 warmup control steps, the same initial actor and frozen world model. Values below are medians of two "
            "updates; setup/checkpoint I/O and warmup are separate from per-update timing.", "",
            "`supermimic` is a local SuperMimic-style standing/frozen-model baseline, not a reproduction of the "
            "published 40-clip training recipe. `filter` filters real collection and evaluation, while its imagined "
            "actor update remains unfiltered. `safetymimic` differentiates imagined QPs and deploys its nominal actor.", "",
            "| Metric | No QP | External filter | Differentiable layer |", "|---|---:|---:|---:|",
            *["| " + name + " | " + " | ".join(value) + " |" for name, value in table], "",
            "Times are seconds unless stated. The methods encounter different states because filtered physical actions "
            "change replay; this is an end-to-end comparison rather than an isolated kernel benchmark.", "",
            "Evaluation is a deliberately partial eight-throw development smoke test, not the full held-out benchmark. "
            "Batch-eight latency is not single-robot latency. The separate batch-one filter evaluation contains only "
            "one complete throw. Check its latency against the 20 ms control interval.", "",
            "Failed finite relaxed QP actions are executed for simulation diagnosis and remain flagged. "
            "Physical contacts/falls and failed filter checks are separate metrics; this experiment supplies no physical safety guarantee.", ""]
    text.extend(["| Evaluation metric | No QP | External filter | Differentiable layer |", "|---|---:|---:|---:|"])
    for name, get in (
        ("controller batch p50 (s)", lambda e: e["controller_batch_seconds_p50"]),
        ("controller batch p95 (s)", lambda e: e["controller_batch_seconds_p95"]),
        ("filter failure rate", lambda e: e["filter_failure_rate"]),
        ("contact rate", lambda e: e["rates"]["contact"]["rate"]),
        ("fall rate", lambda e: e["rates"]["fall"]["rate"]),
    ):
        values_ = [get(record["modes"][mode]["evaluation"]) for mode in columns]
        text.append("| " + name + " | " + " | ".join("n/a" if v is None else f"{v:.4f}" for v in values_) + " |")
    text.extend(["", "Historical reference: the unofficial upstream author reports 14,000 iterations in 1.47 hours "
                 "on one RTX PRO 6000 (0.378 s/iteration average). Its task, batch, precision, model fitting and compilation "
                 "differ, so it is not the denominator for the measured overhead ratios above.", "",
                 "Source: https://github.com/zita-ch/supermimic_vibed#result-lafan1-40-clips-one-rtx-pro-6000-blackwell", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    args.output.with_suffix(".md").write_text("\n".join(text))
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    main()
