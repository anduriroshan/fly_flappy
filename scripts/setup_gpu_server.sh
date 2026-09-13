#!/usr/bin/env bash
# Bare-metal GPU box bootstrap. Idempotent — safe to re-run. Works on Ubuntu
# 22.04 (Python 3.10) or 24.04 (Python 3.12) — deliberately does NOT hardcode
# a Python minor version, since we don't control which base image the rental
# host uses. Both versions have full wheel availability for everything in
# requirements-gpu.txt (torch, pygame, etc.) — the "no wheel yet" problem only
# bites brand-new Python releases (3.13+) that no Ubuntu LTS ships by default.
#
# Assumes NVIDIA driver + CUDA userland already installed by the host image
# (most cloud GPU AMIs ship them). Verify with `nvidia-smi` before running.
set -euo pipefail

echo "[setup] verifying GPU driver..."
if ! command -v nvidia-smi >/dev/null; then
    echo "  nvidia-smi missing. Install the NVIDIA driver first."
    exit 1
fi
nvidia-smi | head -n 6

PY=python3
echo "[setup] system Python: $($PY --version)"

echo "[setup] apt packages..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-pip python3-venv \
    xvfb x11-utils \
    ffmpeg \
    libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1 \
    git tmux htop

echo "[setup] python venv..."
# Use whatever python3 the OS actually ships (see header comment) rather than
# a hardcoded minor version that might not exist as an apt package at all.
$PY -m venv .venv
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
