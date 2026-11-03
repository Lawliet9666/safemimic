"""Report current full-policy experiments only; older residual results are rejected."""

import argparse
import csv
import html
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from safety_mimic.evaluation import dump_json, paired_difference, rate_summary

OBJECTIVE_FORMAT = "safetymimic.supermimic_correction_slack.v1"


def read(path):
    return json.loads(path.read_text())


def link(path, output, label):
    return '<a href="' + html.escape(os.path.relpath(path, output)) + '">' + html.escape(label) + "</a>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--training", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root / "outputs/results"):
        raise ValueError("Reports must be inside outputs/results")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    table, records, sections = [], [], []
    runs = []
    for directory in args.runs:
        config = read(directory / "config.json")
        if config.get("format") != "safetymimic.evaluation.v2":
            raise ValueError(f"Not a current full-policy evaluation: {directory}")
        controller = config["controller"]
        if controller["method"] != "pretrained" and (
            controller["training"].get("imagined_rollouts") != 1
            or controller["training"].get("objective") != OBJECTIVE_FORMAT
            or controller["training"].get("terminal_critic") is not True
        ):
            raise ValueError(f"Not a current SuperMimic/correction/slack policy evaluation: {directory}")
        episodes = read(directory / "episodes.json")
        ids = [episode["throw_id"] for episode in episodes]
        if len(ids) != len(set(ids)) or len(ids) > config["requested_throws"]:
            raise ValueError("Duplicate or excess throw coverage")
        rates = rate_summary(episodes, random_sample=config["sampling"] == "stratified_random")
        complete = (directory / "COMPLETE.json").exists() and len(ids) == config["requested_throws"]
        rate = rates["physical_task_success"]
        row = {
            "label": config["label"],
            "method": config["controller"]["method"],
            "seed": config["controller"]["training"].get("seed"),
            "updates": config["controller"]["training"].get("updates", 0),
            "objective": config["controller"]["training"].get("objective", "pretrained"),
            "split": config["split"],
            "successes": rate["count"],
            "n": rate["n"],
            "success_rate": rate["rate"],
            "ci95_low": rate["ci95"][0] if rate["ci95"] else None,
            "ci95_high": rate["ci95"][1] if rate["ci95"] else None,
            "contacts": rates["contact"]["count"],
            "falls": rates["fall"]["count"],
            "unobserved_outcomes": rates["unobserved_outcomes"],
            "complete_requested": complete,
            "full_manifest": len(ids) == config["manifest_throws"],
            "uses_qp": False,
            "uses_world_model": False,
        }
        table.append(row)
        records.extend({"label": config["label"], **episode} for episode in episodes)
        runs.append((directory, config, episodes))
    table.sort(key=lambda row: (row["method"] != "safetymimic", row["label"]))
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    dump_json(output / "episodes.json", records)
    dump_json(output / "comparison.json", table)
    paired = []
    for i, (_, a, ae) in enumerate(runs):
        for _, b, be in runs[i + 1 :]:
            if (
                a["manifest_sha256"] == b["manifest_sha256"]
                and {e["throw_id"] for e in ae} == {e["throw_id"] for e in be}
                and ae
            ):
                paired.append({"first": a["label"], "second": b["label"], **paired_difference(ae, be)})
    dump_json(output / "paired_comparisons.json", paired)
    for directory, config, episodes in runs:
        label = config["label"]
        videos = output / "videos" / label
        clips = read(videos / "index.json") if (videos / "index.json").exists() else {}
        cards = []
        for episode in episodes:
            throw_id = episode["throw_id"]
            page = output / "episodes" / f"{label}__{throw_id}.html"
            clip = clips.get(throw_id)
            video = (
                ""
                if not clip
                else '<video controls width="800" src="'
                + html.escape(os.path.relpath(videos / clip["normal"], page.parent))
                + '"></video>'
            )
            page.write_text(
                '<!doctype html><meta charset="utf-8"><a href="../index.html">Report</a>'
                + video
                + "<pre>"
                + html.escape(json.dumps(episode, indent=2))
                + "</pre>"
            )
            if clip:
                cards.append(
                    "<p>"
                    + html.escape(throw_id)
                    + " · "
                    + link(page, output, "episode record")
                    + '</p><video controls width="800" src="'
                    + html.escape(os.path.relpath(videos / clip["normal"], output))
                    + '"></video>'
                )
        sections.append(
            "<h2>"
            + html.escape(label)
            + "</h2>"
            + "".join(cards)
            + "<p>"
            + link(directory / "episodes.json", output, "All episode records")
            + "</p>"
        )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        range(len(table)),
        [100 * (r["success_rate"] or 0) for r in table],
        color=["#0055b8" if r["method"] == "safetymimic" else "#9baabd" for r in table],
    )
    ax.set_xticks(range(len(table)), [r["label"] for r in table], rotation=15, ha="right")
    ax.set(ylabel="Physical task success (%)", ylim=(0, 100), title="Policy-only execution")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / "figures" / f"success.{suffix}")
    plt.close(fig)
    training_checks = []
    if args.training:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        for directory in args.training:
            config = read(directory / "config.json")
            # The trainer marks policy-only collection/deployment explicitly.
            if not (config.get("objective") == OBJECTIVE_FORMAT and config.get("terminal_critic") is True
                    and config.get("imagined_rollouts") == 1 and config.get("collection_uses_qp") is False
                    and config.get("deployment_uses_qp") is False):
                raise ValueError("Old training runs cannot be included in a current report")
            rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            for axis, key in zip(axes, ("task", "correction", "qp_slack", "critic_loss")):
                axis.plot(
                    [row["wall_seconds"] / 3600 for row in rows],
                    [row[key] for row in rows],
                    label=directory.name,
                )
                axis.set(xlabel="Training wall time (h)", ylabel=key.replace("_", " "))
            checks = read(directory / "checks.json") if (directory / "checks.json").exists() else None
            training_checks.append(
                {"run": str(directory), "checks": checks, "complete": (directory / "COMPLETE.json").exists()}
            )
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "figures" / "training.png")
        plt.close(fig)
    dump_json(output / "training_checks.json", training_checks)
    rows_html = "".join(
        "<tr>"
        + "".join(
            "<td>" + html.escape(str(row[key])) + "</td>"
            for key in (
                "label",
                "successes",
                "n",
                "success_rate",
                "contacts",
                "falls",
                "unobserved_outcomes",
                "complete_requested",
                "full_manifest",
            )
        )
        + "</tr>"
        for row in table
    )
    body = '<!doctype html><meta charset="utf-8"><title>SafetyMimic full-policy results</title>'
    body += "<style>body{font:16px sans-serif;max-width:1100px;margin:40px auto}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}video,img{max-width:100%}</style>"
    body += "<h1>SafetyMimic: full-policy learning, QP-free execution</h1>"
    body += "<p>Success means no contact, no fall, and recovery within 0.35 m after 3 s. Incomplete outcomes count as task failures; contact and fall rates retain their observed denominators. Partial manifest coverage is labeled explicitly.</p>"
    body += "<table><tr>" + "".join(
        "<th>" + key + "</th>"
        for key in (
            "Run",
            "Successes",
            "N",
            "Rate",
            "Contacts",
            "Falls",
            "Unobserved",
            "Requested complete",
            "Full manifest",
        )
    )
    body += "</tr>" + rows_html + "</table>" + "".join(sections)
    body += '<h2>Comparison</h2><img src="figures/success.png">'
    if args.training:
        body += '<h2>Training diagnostics</h2><img src="figures/training.png">'
    body += "<p>Random validation throws have 95% intervals in comparison.csv. Fixed grids use descriptive counts. Results are reported per training seed; throws do not replace independent training seeds. Smoke checks establish pipeline operation, not learning performance.</p>"
    body += (
        "<p>"
        + " · ".join(
            link(output / name, output, name)
            for name in (
                "comparison.csv",
                "comparison.json",
                "paired_comparisons.json",
                "training_checks.json",
            )
        )
        + "</p>"
    )
    (output / "index.html").write_text(body)
    print(output / "index.html")


if __name__ == "__main__":
    main()
