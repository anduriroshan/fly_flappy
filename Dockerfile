# ============================================================
# Fly-Flappy — headless GPU training image
# Base: CUDA 12.1 + cuDNN 9 (matches requirements-gpu.txt torch build)
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:99

# System deps + headless rendering (Xvfb, ffmpeg for MP4 muxing, OpenGL libs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        git build-essential \
        xvfb x11-utils \
        ffmpeg \
        libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3     /usr/local/bin/pip

WORKDIR /workspace/fly_flappy

# Requirements first so Docker caches the pip layer across code changes.
COPY requirements-gpu.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-gpu.txt

COPY . .

# Entrypoint script boots Xvfb, then execs whatever CMD was passed.
COPY docker/entrypoint.sh /usr/local/bin/fly-entrypoint
RUN chmod +x /usr/local/bin/fly-entrypoint

ENTRYPOINT ["fly-entrypoint"]
CMD ["python", "-m", "scripts.train", "--profile", "full"]
