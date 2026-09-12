#!/usr/bin/env bash
# Bare-metal Ubuntu 22.04 GPU box bootstrap. Idempotent — safe to re-run.
# Assumes NVIDIA driver + CUDA userland already installed by the host image
# (most cloud GPU AMIs ship them). Verify with `nvidia-smi` before running.
set -euo pipefail

echo "[setup] verifying GPU driver..."
if ! command -v nvidia-smi >/dev/null; then
    echo "  nvidia-smi missing. Install the NVIDIA driver first."
    exit 1
fi
nvidia-smi | head -n 6

echo "[setup] apt packages..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    xvfb x11-utils \
    ffmpeg \
    libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1 \
    git tmux htop

echo "[setup] python venv..."
python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt

echo "[setup] smoke test on GPU..."
python -m scripts.smoke_test

cat <<'EOF'

[setup] complete. To train:
    source .venv/bin/activate
    # Xvfb only needed if you use pixel observations; simple obs mode
    # does not require it.
    Xvfb :99 -screen 0 1280x720x24 &
    export DISPLAY=:99
    python -m scripts.train --profile full

Recommended: run inside tmux so a dropped SSH doesn't kill the run.
EOF
