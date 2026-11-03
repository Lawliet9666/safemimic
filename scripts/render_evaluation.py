"""Two-view, time-aligned diagnostic clips from recorded simulator poses only."""

import argparse
import csv
import html
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(ROOT / "vendor/supermimic/docker/10_nvidia.json"))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from safety_mimic.evaluation import dump_json, sha256


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def timeline(rows, episode, width=960, height=200):
    canvas = Image.new("RGB", (width, height), "#f2f4f7")
    draw = ImageDraw.Draw(canvas)
    panels = [
        ("clearance (m)", ["clearance_min_m"]),
        ("root height (m)", ["root_height_m"]),
        ("upright cosine", ["root_tilt_cos"]),
        ("action norm", ["action_norm"]),
    ]
    for panel, (title, keys) in enumerate(panels):
        x0, x1 = panel * width // 4 + 12, (panel + 1) * width // 4 - 12
        y0, y1 = 30, height - 30
        draw.text((x0, 5), title, fill="#182536", font=font(11))
        values = [[float(r[k]) for r in rows if r.get(k) not in (None, "")] for k in keys]
        all_values = [v for series in values for v in series if np.isfinite(v)]
        low, high = (min(0, min(all_values)), max(0, max(all_values))) if all_values else (0, 1)
        if high - low < 1e-8:
            high = low + 1
        for key, color in zip(keys, ("#2864dc", "#d34535")):
            points = [
                (
                    x0 + float(r["time_end"]) / 3 * (x1 - x0),
                    y1 - (float(r[key]) - low) / (high - low) * (y1 - y0),
                )
                for r in rows
                if r.get(key) not in (None, "")
            ]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
        zero = y1 - (0 - low) / (high - low) * (y1 - y0)
        draw.line((x0, zero, x1, zero), fill="#aeb6c0")
        draw.text((x0, y1 + 5), "0 s", fill="#182536", font=font(11))
        draw.text((x1 - 20, y1 + 5), "3 s", fill="#182536", font=font(11))
        draw.text((x0, 17), f"[{low:.3g}, {high:.3g}]", fill="#677386", font=font(10))
        for event, color in (("fall", "#df9500"), ("contact", "#d34535")):
            if event in episode["first_events"]:
                x = x0 + episode["first_events"][event] / 3 * (x1 - x0)
                draw.line((x, y0, x, y1), fill=color, width=1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--throw_ids", nargs="*")
    args = parser.parse_args()
    start = time.monotonic()
    config = json.loads((args.run / "config.json").read_text())
    episodes = {e["throw_id"]: e for e in json.loads((args.run / "episodes.json").read_text())}
    selection = json.loads((args.run / "selected_clips.json").read_text())
    ids = args.throw_ids or [row["throw_id"] for row in selection]
    geometry = json.loads((args.run / "geometry.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "upstream_renderer", ROOT / "vendor/supermimic/tools/render.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    xml = upstream.two_robot_xml().replace(
        "</worldbody>",
        '<body name="ball"><freejoint/><geom type="sphere" size="0.08" rgba="0.9 0.1 0.08 1" '
        'contype="0" conaffinity="0"/></body></worldbody>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.headlight.ambient[:] = 0.45
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 360, 480)
    reference = np.loadtxt(ROOT / "data/standing/stand.csv", delimiter=",")[0]
    reference_q = np.concatenate((reference[:3], reference[6:7], reference[3:6], reference[7:]))
    args.output.mkdir(parents=True, exist_ok=True)
    previous_cost = (
        json.loads((args.output / "render_cost.json").read_text())
        if (args.output / "render_cost.json").exists()
        else {}
    )
    index = (
        json.loads((args.output / "index.json").read_text()) if (args.output / "index.json").exists() else {}
    )
    original_colors = model.geom_rgba.copy()

    def link_evidence(throw_id, episode):
        sidecar = args.output / (throw_id + ".episode.json")
        dump_json(sidecar, episode)
        key = args.output.name
        index[throw_id].update(
            episode_record=sidecar.name,
            episode_page=os.path.relpath(
                args.output.parents[1] / "episodes" / (key + "__" + throw_id + ".html"), args.output
            ),
            trajectory_sha256=sha256(args.run / episode["trajectory"]),
            video_sha256=sha256(args.output / index[throw_id]["normal"]),
        )
        dump_json(args.output / "index.json", index)

    for throw_id in ids:
        normal = args.output / (throw_id + ".mp4")
        slow = args.output / (throw_id + "_slow.mp4")
        episode = episodes[throw_id]
        if normal.exists() and slow.exists() and throw_id in index:
            link_evidence(throw_id, episode)
            continue
        trajectory = np.load(args.run / episode["trajectory"])
        csv_path = args.run / "trajectories" / (throw_id + ".csv")
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        background = timeline(rows, episode)
        encoder = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "960x660",
                "-r",
                "50",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-threads",
                "6",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                str(normal),
            ],
            stdin=subprocess.PIPE,
        )
        frames = len(trajectory["root_p"])
        for display_frame in range(151):
            frame = min(display_frame, frames - 1)
            t = frame * 0.02
            model.geom_rgba[:] = original_colors
            data.qpos[:36] = np.concatenate(
                (trajectory["root_p"][frame], trajectory["root_quat"][frame], trajectory["q"][frame])
            )
            data.qpos[36:72] = reference_q
            data.qpos[72:75], data.qpos[75:79] = trajectory["ball_p"][frame], [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            contacts = (
                episode["contact_links"] if t >= episode["first_events"].get("contact", float("inf")) else []
            )
            for link in contacts:
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
                if body >= 0:
                    model.geom_rgba[model.geom_bodyid == body] = [0.95, 0.15, 0.08, 1]
            threat = int(trajectory["clearances"][frame].argmin())
            image = Image.new("RGB", (960, 660), "white")
            for view in range(2):
                camera = mujoco.MjvCamera()
                camera.azimuth, camera.elevation = (135, -12) if view == 0 else (45, -18)
                root = trajectory["root_p"][frame]
                if view == 0:
                    camera.lookat[:], camera.distance = [root[0], root[1], 0.8], 4.2
                else:
                    ball = trajectory["ball_p"][frame]
                    center = (np.array([root[0], root[1], 0.8]) + ball) / 2
                    span = np.linalg.norm(ball - np.array([root[0], root[1], 0.8]))
                    camera.lookat[:], camera.distance = center, max(4.2, 2.0 * span + 2.0)
                renderer.update_scene(data, camera=camera)
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([geometry["radii"][threat]] * 3),
                    trajectory["geom_centers"][frame, threat],
                    np.eye(3).flatten(),
                    np.array([1, 0.5, 0.05, 0.23], dtype=np.float32),
                )
                renderer.scene.ngeom += 1
                image.paste(Image.fromarray(renderer.render()), (view * 480, 100))
            draw = ImageDraw.Draw(image)
            observed = episode["physical_outcome_observed"]
            task = ("PASS" if episode["physical_task_success"] else "FAIL") if observed else "UNOBSERVED"
            draw.text(
                (770, 9),
                "Final task: " + task,
                fill="#21663b" if task == "PASS" else "#a33216",
                font=font(12),
            )
            draw.text(
                (770, 30),
                "Execution: policy only",
                fill="#596579",
                font=font(12),
            )
            label = config["label"]
            checkpoint = Path(config["controller"].get("checkpoint") or "pretrained SuperMimic").name
            draw.text(
                (12, 8), f"SafetyMimic | {label} | {throw_id} | t={t:.3f} s", fill="#152331", font=font(17)
            )
            draw.text(
                (12, 32),
                f"checkpoint: {checkpoint} | target: {episode['region']} | flight: {episode['flight_time_bin']} s",
                fill="#152331",
                font=font(13),
            )
            status = []
            for event in ("contact", "fall"):
                if episode["first_events"].get(event, float("inf")) <= t:
                    status.append(f"{event} @ {episode['first_events'][event]:.3f}s")
            if display_frame >= frames and not episode["completed"]:
                status.append("RECORDING STOPPED: outcome unobserved")
            draw.text(
                (12, 53),
                " | ".join(status) or "No failure observed yet",
                fill="#a33216" if status else "#21663b",
                font=font(14),
            )
            draw.text(
                (12, 78),
                f"Orange sphere: nearest proxy ({geometry['links'][threat]}); red links: recorded contact",
                fill="#596579",
                font=font(12),
            )
            panel = background.copy()
            d = ImageDraw.Draw(panel)
            for j in range(4):
                x = j * 240 + 12 + t / 3 * (240 - 24)
                d.line((x, 30, x, 170), fill="black", width=2)
            image.paste(panel, (0, 460))
            encoder.stdin.write(np.asarray(image).tobytes())
            if display_frame == min(35, frames - 1):
                image.save(args.output / (throw_id + ".png"))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
        event = min(episode["first_events"].values(), default=episode["flight_time_bin"])
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(max(0, event - 0.35)),
                "-t",
                "0.7",
                "-i",
                str(normal),
                "-vf",
                "setpts=4*(PTS-STARTPTS)",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "6",
                "-pix_fmt",
                "yuv420p",
                str(slow),
            ],
            check=True,
        )
        index[throw_id] = {
            "normal": normal.name,
            "slow": slow.name,
            "episode_run": str(args.run),
            "throw_id": throw_id,
            "recorded_frames": frames,
            "fps": 50,
            "recorded_duration": (frames - 1) * 0.02,
            "padding_frozen": not episode["completed"],
            "reasons": next((s["reasons"] for s in selection if s["throw_id"] == throw_id), ["comparison"]),
        }
        link_evidence(throw_id, episode)
        print(json.dumps({"rendered": throw_id, "label": config["label"]}), flush=True)
    renderer.close()
    listing = '<!doctype html><meta charset="utf-8"><title>SafetyMimic episode clips</title><p><a href="../../index.html">Evaluation report</a></p><ul>'
    for throw_id, record in index.items():
        listing += (
            "<li>"
            + html.escape(throw_id)
            + ": "
            + " · ".join(
                '<a href="' + html.escape(record[field]) + '">' + label + "</a>"
                for field, label in (
                    ("normal", "Video"),
                    ("slow", "Slow motion"),
                    ("episode_page", "Episode evidence"),
                    ("episode_record", "Episode JSON"),
                )
            )
            + "</li>"
        )
    (args.output / "index.html").write_text(listing + "</ul>")
    dump_json(
        args.output / "render_cost.json",
        {
            "last_invocation_seconds": time.monotonic() - start,
            "cumulative_function_seconds": previous_cost.get(
                "cumulative_function_seconds", previous_cost.get("last_invocation_seconds", 0)
            )
            + time.monotonic()
            - start,
            "invocations": previous_cost.get("invocations", 0) + 1,
            "videos_indexed": len(index),
            "rendering_uses_recorded_poses": True,
        },
    )


if __name__ == "__main__":
    main()
