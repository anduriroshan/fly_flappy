# GPU Server Runbook

Sequential commands to go from a fresh Vast.ai box to full training +
morphology render. Run everything below **on the GPU server** (SSH in
first) unless a step says "on laptop."

Target hardware this was written against: 2x GPU (16GB+ VRAM each),
32 vCPUs, 100GB+ disk. GPU 0 = training, GPU 1 = Blender render.

---

## 0. On your laptop — confirm everything is pushed

```bash
cd /path/to/fly_flappy
git status              # should say "working tree clean"
git push origin main    # if anything was uncommitted
```

If this isn't clean, stop and commit/push first — the server clone pulls
whatever's on `origin/main`, including nothing you haven't pushed yet.

---

## 1. SSH into the server and check hardware

```bash
ssh -p <PORT> root@<HOST>

nvidia-smi -L            # confirm both GPUs show up, note their index order
nproc                     # confirm vCPU count
free -h                   # confirm RAM
df -h .                   # confirm disk
python3 --version         # note the version (affects pygame wheel availability)
```

---

## 2. Install Git LFS, then clone

The repo's connectome CSVs (`data/connectome/raw/*.csv`) and training
videos (`runs/videos/*.mp4`) are tracked via Git LFS. Install LFS
**before** cloning, or you'll get tiny pointer-file stubs instead of the
real ~222MB of CSV data.

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
pip install -r requirements-gpu.txt
```

---

## 4. Verify before doing anything expensive

```bash
python -m pytest tests -q          # 18 tests — includes the gradcheck for
                                    # the sparse-op OOM fix; must all pass
python -m scripts.smoke_test        # synthetic data, fast, proves the
                                    # whole pipeline end-to-end on this box
```

If either of these fails, stop and fix it before touching real data or
GPUs — everything downstream assumes this passes.

---

## 5. GPU assignment

Two GPUs, one job each. `CUDA_VISIBLE_DEVICES` restricts what each
process can see at the driver level — no code changes needed, it's
inherited by subprocesses too (including the Blender subprocess that
`render_morphology.py` launches).

- **GPU 0** → RL training (`scripts/train.py`)
- **GPU 1** → Blender/Cycles morphology render (`scripts/render_morphology.py`)

---

## 6. Calibration run — do this before committing to the full run

The full profile trains 5,000,000 timesteps on the real 166,700-neuron
connectome. Don't guess how long that takes — measure it on THIS
hardware first.

```bash
tmux new -s calibrate
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --profile full
```

Watch the **first rollout's printed `fps`** in the SB3 console output
(appears after the first 8,192 timesteps: `n_steps=512 * n_envs=16`).
Do the math: `5,000,000 / fps` seconds = real estimated wall-clock time.

- If that number looks acceptable: `Ctrl+B` then `D` to detach, let it
  keep running (Phase 7 picks up the same running session).
- If it's too slow: `Ctrl+C` to stop, then lower `budgets.full.neurons`
  in `config/config.yaml` (e.g. to 30,000 or 20,000 — snowball sampling
  keeps local synapse density realistic at smaller scale) and re-run
  this step.

---

## 7. Full training (if not already left running from Phase 6)

```bash
tmux attach -t calibrate     # reattach if you detached earlier
# — or, if starting fresh: —
tmux new -s train
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --profile full 2>&1 | tee runs/train_full.log
```

Detach (`Ctrl+B`, `D`) any time — it keeps running. Reattach with
`tmux attach -t train` (or `calibrate`, whichever session name you used).

Output: `runs/checkpoints/ppo_connectome_full.zip` (~429MB — bakes in the
full sparse synapse graph, this is normal) and a live 3-panel dashboard
video under `runs/videos/`.

---

## 8. Blender setup (GPU 1, any time — doesn't need to wait for training)

Skeletons are fetched from Janelia's PUBLIC GCS bucket (CC-BY, no account
or token needed) — so the only real prerequisite here is Blender itself.
`requirements-render.txt` (navis + neuprint-python) is only needed if you
opt into `--skeleton-source neuprint` for neuropil meshes; skip it otherwise.

`setup_blender.sh` runs `apt-get install` for a handful of X11/graphics
shared libraries (`libsm6`, `libxext6`, etc.) that Blender's binary is
dynamically linked against — needed even in headless `--background` mode
with no display, since the dynamic linker resolves them at process
startup regardless of whether the display code path ever runs. This is
automatic now; you'll just see apt output scroll by.

```bash
bash scripts/setup_blender.sh
export BLENDER_BIN=$(cat .blender_bin)

# confirm GPU 1 is visible to Cycles specifically
CUDA_VISIBLE_DEVICES=1 $BLENDER_BIN -b --python-expr "
import bpy
p = bpy.context.preferences.addons['cycles'].preferences
p.compute_device_type = 'OPTIX'
p.get_devices()
print([d.name for d in p.devices])
"
```

---

## 9. Morphology render (once training has produced a checkpoint)

No neuprint token needed — skeletons come from the public GCS bucket by default.

```bash
tmux new -s render
source .venv/bin/activate
export BLENDER_BIN=$(cat .blender_bin)

# record real connectome activation over a rollout (CPU-cheap, no GPU needed)
CUDA_VISIBLE_DEVICES=1 python -m scripts.record_activation --profile full \
    --checkpoint runs/checkpoints/ppo_connectome_full.zip \
    --out runs/morphology/activation.npz --steps 900 --max-neurons 400

# fail-fast check: renders ONE frame, catches GPU/skeleton/material issues early
# (fetches real skeletons from the public bucket, then renders one still)
CUDA_VISIBLE_DEVICES=1 python -m scripts.render_morphology \
    --recording runs/morphology/activation.npz --skip-neuropil --dry-run

# full render (only after the dry-run looks right)
CUDA_VISIBLE_DEVICES=1 python -m scripts.render_morphology \
    --recording runs/morphology/activation.npz --skip-neuropil \
    --out runs/videos/morphology.mp4
```

`--skip-neuropil` avoids the "neuropil meshes aren't in the public bucket"
notice. If you later get a working neuprint token and want the translucent
brain backdrop meshes, drop `--skip-neuropil` and add
`--skeleton-source neuprint` (needs `pip install -r requirements-render.txt`
+ `export NEUPRINT_TOKEN=...`).

---

## 10. Getting results back to your laptop

Checkpoints and morphology outputs are gitignored (only `runs/videos/*.mp4`
and the connectome CSVs are tracked via LFS). Pick one:

**Videos** — already covered by LFS, just push from the server:
```bash
git add runs/videos/*.mp4
git commit -m "Add training/morphology videos from GPU run"
git push origin main
# then on laptop: git pull
```

**Checkpoints** (not in git) — scp directly:
```bash
# from your LAPTOP, not the server:
scp -P <PORT> root@<HOST>:~/fly_flappy/runs/checkpoints/ppo_connectome_full.zip \
    ./runs/checkpoints/
```

---

## Quick reference — env vars used throughout

| Variable | Value | Why |
|---|---|---|
| `SDL_VIDEODRIVER` | `dummy` | headless pygame rendering, no X server needed |
| `CUDA_VISIBLE_DEVICES` | `0` (train) / `1` (Blender) | pins each workload to one physical GPU |
| `BLENDER_BIN` | `$(cat .blender_bin)` | set by `scripts/setup_blender.sh` |
| `NEUPRINT_TOKEN` | your account token | OPTIONAL — only for `--skeleton-source neuprint` (neuropil meshes). Default skeleton fetch uses the public bucket, no token. |
