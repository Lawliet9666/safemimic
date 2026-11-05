import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_hpc_shell_syntax():
    for script in [*ROOT.glob("scripts/hpc/*.sh"), ROOT / "scripts/hpc/train_ou.sbatch",
                   *ROOT.glob("tests/integration/*.sbatch")]:
        subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("resume,preflight", [(False, False), (True, False), (False, True)])
def test_launch_requests_exactly_two_workers_and_preserves_gpu_assignment(tmp_path, resume, preflight):
    executable = tmp_path / "python_mock"
    calls = tmp_path / "calls.jsonl"
    executable.write_text(f"#!{sys.executable}\n" +
                          "import json, os, sys\n" +
                          "with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps({"
                          "'args':sys.argv[1:], 'devices':os.environ['CUDA_VISIBLE_DEVICES']})+'\\n')\n")
    executable.chmod(0o755)
    environment = {**os.environ, "OU_PYTHON": str(executable), "CALLS": str(calls),
                   "SLURM_JOB_ID": "test", "CUDA_VISIBLE_DEVICES": "GPU-assigned-A,GPU-assigned-B",
                   "RUN_DIR": str(tmp_path / "run"), "STOP_FILE": str(tmp_path / "stop"),
                   "AVOIDANCE_WEIGHT": "12.5",
                   "ADAPTIVE_GAIN": "1",
                   "PREFLIGHT_ONLY": "1" if preflight else "0"}
    if resume:
        environment["RESUME"] = "outputs/runs/old/checkpoints/latest.json"
    else:
        environment.pop("RESUME", None)
    subprocess.run(["bash", str(ROOT / "scripts/hpc/launch.sh")], cwd=tmp_path, env=environment, check=True)
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(records) == 2
    assert all(record["devices"] == environment["CUDA_VISIBLE_DEVICES"] for record in records)
    command = records[1]["args"]
    assert command[:2] == ["-m", "torch.distributed.run"]
    assert "--nproc-per-node=2" in command
    if preflight:
        assert "tests/integration/preflight.py" in command
        assert (ROOT / "tests/integration/preflight.py").is_file()
        return
    assert command[command.index("--runtime") + 1] == "slurm"
    if resume:
        assert "--resume" in command and "--actor_lr" not in command
    else:
        assert command[command.index("--gradient_accumulation_steps") + 1] == "2"
        assert command[command.index("--ball_lr") + 1] == "5e-5"
        assert command[command.index("--baseline_avoidance_weight") + 1] == "12.5"
        assert "--adaptive_gain" in command


def test_submission_requires_a_real_partition_name(tmp_path):
    environment = dict(os.environ)
    environment.pop("OU_PARTITION", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/hpc/submit_ou.sh")], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode != 0 and "OU_PARTITION" in result.stderr


def test_batch_apptainer_passes_slurm_gpu_assignment_explicitly(tmp_path):
    srun = tmp_path / "srun"
    calls = tmp_path / "srun.json"
    srun.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    "with open(os.environ['CALLS'], 'w') as f: json.dump(sys.argv[1:], f)\n")
    srun.chmod(0o755)
    image = tmp_path / "simulation.sif"
    image.touch()
    environment = {**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"], "CALLS": str(calls),
                   "SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_JOB_ID": "test", "SLURM_CPUS_PER_TASK": "32",
                   "SAFETYMIMIC_SIF": str(image), "CUDA_VISIBLE_DEVICES": "GPU-A,GPU-B"}
    environment.pop("RUN_DIR", None)
    subprocess.run(["bash", str(ROOT / "scripts/hpc/train_ou.sbatch")], env=environment, check=True)
    command = json.loads(calls.read_text())
    assert command[:2] == ["--ntasks=1", "--cpu-bind=cores"]
    assert "--nv" in command and "CUDA_VISIBLE_DEVICES=GPU-A,GPU-B" in command


def test_submission_from_another_directory_uses_project_submit_directory(tmp_path):
    sbatch = tmp_path / "sbatch"
    calls = tmp_path / "submission.json"
    sbatch.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                      "with open(os.environ['CALLS'], 'w') as f: json.dump({'cwd':os.getcwd(), 'args':sys.argv[1:]}, f)\n")
    sbatch.chmod(0o755)
    environment = {**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"], "CALLS": str(calls),
                   "OU_PARTITION": "assigned_queue", "OU_ACCOUNT": "assigned_account"}
    subprocess.run(["bash", str(ROOT / "scripts/hpc/submit_ou.sh"), "--time=00:15:00"],
                   cwd=tmp_path, env=environment, check=True)
    record = json.loads(calls.read_text())
    assert record["cwd"] == str(ROOT)
    assert "--partition=assigned_queue" in record["args"] and "--account=assigned_account" in record["args"]
    assert record["args"][-1] == str(ROOT / "scripts/hpc/train_ou.sbatch")


def test_matilda_request_selects_typed_gpu6000_resources(tmp_path):
    sbatch = tmp_path / "sbatch"
    calls = tmp_path / "submission.json"
    sbatch.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                      "with open(os.environ['CALLS'], 'w') as f: json.dump(sys.argv[1:], f)\n")
    sbatch.chmod(0o755)
    environment = {**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"],
                   "CALLS": str(calls), "OU_PARTITION": "gpu6000-short"}
    environment.pop("OU_GRES", None)
    subprocess.run(["bash", str(ROOT / "scripts/hpc/submit_ou.sh")], env=environment, check=True)
    assert "--gres=gpu:6000:2" in json.loads(calls.read_text())


def test_validation_and_training_share_safety_defaults():
    import argparse
    from safety_mimic.settings import SAFETY_DEFAULTS, add_safety_arguments

    parser = argparse.ArgumentParser()
    add_safety_arguments(parser)
    assert vars(parser.parse_args([])) == SAFETY_DEFAULTS
    args = parser.parse_args(["--velocity_time", "0.4", "--anticipation_mode", "all"])
    assert args.velocity_time == 0.4 and args.anticipation_mode == "all"
