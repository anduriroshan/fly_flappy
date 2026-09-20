# GPU Server Runbook

Sequential commands to go from a fresh Vast.ai (or similar) box to a
trained PPO checkpoint. Run everything below **on the GPU server** (SSH
in first) unless a step says "on laptop." Training is the only thing
the GPU box is for — everything downstream (web platform, morphology
visualization) runs locally on the laptop with just the checkpoint.

Target hardware: 1x GPU (16GB+ VRAM), 16+ vCPUs, 50GB+ disk.

---

## 0. On your laptop — confirm everything is pushed

```bash
cd /path/to/fly_flappy
git status              # should say "working tree clean"
git push origin main    # if anything was uncommitted
```

The server clone pulls whatever's on `origin/main`, including nothing
you haven't pushed yet.

---

## 1. SSH into the server and check hardware

```bash
ssh -p <PORT> root@<HOST>

nvidia-smi -L            # confirm the GPU shows up
nproc                     # confirm vCPU count
free -h                   # confirm RAM
df -h .                   # confirm disk
python3 --version         # note the version
```

---

## 2. Install Git LFS, then clone

The repo's connectome CSVs (`data/connectome/raw/*.csv`) are tracked via
Git LFS. Install LFS **before** cloning, or you'll get tiny pointer-file
stubs instead of the real ~222MB of CSV data.

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install -y git-lfs
git lfs install

git clone https://github.com/anduriroshan/fly_flappy.git
cd fly_flappy
ls -la data/connectome/raw/     # should show real file sizes (~204M + ~18M), not tiny pointer stubs
```

---

## 3. Python environment

```bash
export SDL_VIDEODRIVER=dummy    # avoids a headless pygame render failure —
                                 # add this to ~/.bashrc so it's set in every new shell

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Check the driver's actual CUDA ceiling BEFORE installing torch:
nvidia-smi   # look at "CUDA Version: XX.X" in the header

# Install torch as its OWN command with `--index-url` (not `--extra-index-url`)
# so pip can ONLY see cu121 wheels and can't silently prefer a newer,
# driver-incompatible build from default PyPI. cu121 runs fine on any driver
# reporting CUDA 12.1 or newer (CUDA is backward compatible) — if `nvidia-smi`
# reports something older than 12.1, use the matching whl/cuXXX index instead.
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Confirm BEFORE installing anything else:
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# torch.cuda.is_available() must print True. If it prints False, stop here —
# do not proceed to pytest or training; re-check nvidia-smi vs the torch
# build you just installed.

pip install -r requirements-gpu.txt
```

---

## 4. Verify before doing anything expensive

```bash
python -m pytest tests -q          # includes the gradcheck for the sparse-op
                                    # OOM fix; must all pass
python -m scripts.smoke_test        # synthetic data, fast, proves the
                                    # whole pipeline end-to-end on this box
```

If either fails, fix it before touching real data or the GPU — everything
downstream assumes this passes.

---

## 5. Calibration run — do this before committing to the full run

The full profile trains 5,000,000 timesteps on the (subsampled) real
connectome. Don't guess how long that takes — measure it on THIS
hardware first.

```bash
tmux new -s train
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --profile full 2>&1 | tee runs/train_full.log
```

Watch the **SB3 console output** for the reported `fps` after a few
iterations. Compute expected wall-clock: `5,000,000 / fps` seconds. The
iteration-1 `fps` is misleading (only reflects rollout collection, not
backward pass) — take the marginal rate between two later, consecutive
iterations: `(Δtimesteps)/(Δtime)`.

- If throughput is acceptable: `Ctrl+B` then `D` to detach, let it keep
  running.
- If too slow: `Ctrl+C` to stop, then lower `budgets.full.neurons` in
  `config/config.yaml` (e.g. to 20,000 — snowball sampling keeps local
  synapse density realistic at smaller scale) and re-run.

---

## 6. Full training (if not already left running from Phase 5)

```bash
tmux attach -t train                # reattach if you detached earlier
# — or, if starting fresh: —
tmux new -s train
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --profile full 2>&1 | tee runs/train_full.log
```

Detach any time with `Ctrl+B`, `D` — training keeps running. Reattach
with `tmux attach -t train`. `Ctrl+C` inside the session stops it
cleanly and still saves the checkpoint via the `try/finally` in
[src/rl/train.py](src/rl/train.py) — no lost work on interrupt.

Output: `runs/checkpoints/ppo_connectome_full.zip`. This single file is
everything the web platform needs on the laptop side.

---

## 7. Getting the checkpoint back to your laptop

```bash
# from your LAPTOP, not the server:
scp -P <PORT> root@<HOST>:~/fly_flappy/runs/checkpoints/ppo_connectome_full.zip \
    ./runs/checkpoints/
```

Then **shut down the Vast.ai instance** — billing accrues continuously
while the box is up, not just during active compute.

Back on your laptop, run the web platform:
```bash
python -m scripts.serve
```

---

## Quick reference — env vars used throughout

| Variable | Value | Why |
|---|---|---|
| `SDL_VIDEODRIVER` | `dummy` | headless pygame rendering, no X server needed |
| `CUDA_VISIBLE_DEVICES` | `0` | pins training to one physical GPU (only matters on multi-GPU boxes) |
