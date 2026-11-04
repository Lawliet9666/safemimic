FROM safetymimic-isaaclab:46d6cc3
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*
# Preserve upstream Python package versions; use CUDA 12 variants for the R570 host driver.
RUN uv pip install --python /opt/isaaclab-venv/bin/python --reinstall \
      'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install --python /opt/isaaclab-venv/bin/python --reinstall --no-deps \
      'https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl' \
    && uv pip install --python /opt/isaaclab-venv/bin/python --no-deps qpth==0.0.18 \
    && uv pip install --python /opt/isaaclab-venv/bin/python \
      safetensors==0.8.0 cvxpy==1.9.3 pytest==9.1.1
RUN uv --no-config pip install --python /opt/isaaclab-venv/bin/python --no-deps \
      numpy==2.3.5 packaging==26.0 newton-usd-schemas==0.5.0
COPY vendor/supermimic/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV PYTHONPATH=/workspace/safetymimic/src:/workspace/safetymimic/vendor/supermimic
ENV WARP_CACHE_PATH=/workspace/safetymimic/.cache/warp
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/safetymimic/.cache/inductor
ENV MPLCONFIGDIR=/tmp/safetymimic-mpl
ENV MUJOCO_GL=egl
WORKDIR /workspace/safetymimic

