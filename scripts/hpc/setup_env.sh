#!/usr/bin/env bash
# CPU-only setup; run inside a Slurm allocation. --check reuses an existing runtime.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
user_root="$(cd "$project_root/../.." && pwd)"
environment_root="${OU_ENV_ROOT:-$user_root/envs/safetymimic}"
cd "$project_root"
if [[ "${1:-}" == --check ]]; then
  exec bash scripts/hpc/python.sh tests/integration/check_env.py
fi
: "${SLURM_JOB_ID:?Run setup in a CPU Slurm allocation}"
[[ "$(uname -s)" == Linux ]] || { echo 'The simulator environment is Linux-only' >&2; exit 2; }
command -v conda >/dev/null
command -v git >/dev/null
mkdir -p outputs/setup outputs/logs "$user_root/.cache/conda-pkgs"
export CONDA_PKGS_DIRS="$user_root/.cache/conda-pkgs"
if [[ ! -d "$environment_root/conda-meta" ]]; then
  [[ ! -e "$environment_root" ]] || { echo "Existing non-Conda path: $environment_root" >&2; exit 2; }
  conda create --solver libmamba --override-channels -c conda-forge -p "$environment_root" python=3.12 pip uv=0.12.9 -y
fi
"$environment_root/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,12)'
"$environment_root/bin/python" scripts/bootstrap_sources.py
apptainer="${APPTAINER:-$project_root/.cache/apptainer-public-1.5.1/bin/apptainer}"
if [[ ! -x "$apptainer" ]]; then
  curl -fL --retry 3 -o outputs/setup/install-apptainer.sh https://raw.githubusercontent.com/apptainer/apptainer/v1.5.1/tools/install-unprivileged.sh
  sed 's@https://download.fedoraproject.org/pub/epel@https://dl.fedoraproject.org/pub/epel@g' outputs/setup/install-apptainer.sh > outputs/setup/install-apptainer-direct.sh
  bash outputs/setup/install-apptainer-direct.sh -d el8 -v 1.5.1 "$project_root/.cache/apptainer-public-1.5.1"
fi
export APPTAINER="$apptainer"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$project_root/.cache/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:?Set TMPDIR to writable scratch storage}/apptainer}"
image="${SAFETYMIMIC_RUNTIME_IMAGE:-$project_root/.cache/containers/ubuntu24-safetymimic.sif}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$image")"
if [[ ! -f "$image" ]]; then
  "$apptainer" build --fakeroot "$image" scripts/hpc/ubuntu24.def
fi
export UV_CACHE_DIR="$user_root/.cache/uv-ubuntu24" PYTHONNOUSERSITE=1
export UV_PYTHON="$environment_root/bin/python" UV_PYTHON_PREFERENCE=only-system
runtime=("$apptainer" exec --bind "$user_root:$user_root,$TMPDIR:$TMPDIR" --env "ISAACLAB_PATH=$project_root/vendor/supermimic/IsaacLab" --env "PYTHONPATH=$project_root/src:$project_root/vendor/supermimic" "$image")
uv="$environment_root/bin/uv"
python_executable="$environment_root/bin/python"
cd "$project_root/vendor/supermimic/IsaacLab"
"${runtime[@]}" "$uv" export --frozen --no-dev --no-hashes --extra all --extra importers --extra test --no-emit-project > "$project_root/outputs/setup/isaaclab_requirements.txt"
"${runtime[@]}" "$uv" pip install --python "$python_executable" --index-strategy unsafe-best-match --prerelease allow --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu130 -r "$project_root/outputs/setup/isaaclab_requirements.txt"
"${runtime[@]}" "$uv" pip install --python "$python_executable" --reinstall torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
"${runtime[@]}" "$uv" pip install --python "$python_executable" --reinstall --no-deps 'https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
"${runtime[@]}" "$uv" pip install --python "$python_executable" --no-deps qpth==0.0.18
"${runtime[@]}" "$uv" pip install --python "$python_executable" safetensors==0.8.0 cvxpy==1.9.3 pytest==9.1.1 ruff 'huggingface_hub>=0.33.4,<2'
"${runtime[@]}" "$uv" --no-config pip install --python "$python_executable" --no-deps numpy==2.3.5 packaging==26.0 newton-usd-schemas==0.5.0
cd "$project_root"
"${runtime[@]}" "$uv" pip install --python "$python_executable" --no-deps -e .
ln -sfn "$project_root/scripts/hpc/python.sh" "$environment_root/bin/safetymimic-python"
"${runtime[@]}" "$python_executable" tests/integration/check_env.py
"${runtime[@]}" "$uv" pip freeze --python "$python_executable" > outputs/setup/installed_packages.txt
