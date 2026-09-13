# fly-flappy

Drive a Flappy Bird RL agent with the adult *Drosophila melanogaster*
connectome mapped by the FlyWire consortium (~139k neurons, ~50M
synapses). PPO optimises synaptic weights over a fixed biological
topology; the agent's decision to flap or fall emerges from motor
neurons downstream of visual-system-mapped sensory neurons.

- **Local (this laptop, no GPU)**: `smoke` profile → 2k-neuron synthetic
  connectome, 1 env, ~5k timesteps. Enough to prove the wiring end-to-end.
- **GPU server**: `full` profile → real FlyWire graph, 16 parallel envs,
  5M timesteps, live 3-panel MP4 telemetry, TensorBoard.

## Layout

```
fly_flappy/
├── config/config.yaml         # single source of truth for both profiles
├── src/
│   ├── connectome/            # loader, mapping, PyTorch sparse net
│   ├── env/                   # gym wrapper + vec-env factory
│   ├── rl/                    # SB3 policy, training loop, callbacks
│   ├── telemetry/             # 3-panel dashboard + MP4 recorder
│   └── utils/                 # device/config helpers
├── scripts/                   # entrypoints: train / evaluate / smoke_test / download
├── tests/                     # pytest suites
├── docker/entrypoint.sh       # Xvfb bootstrap
├── Dockerfile
├── docker-compose.yml
├── requirements.txt           # CPU-local (laptop)
└── requirements-gpu.txt       # CUDA 12.1 (server)
```

## Local (Windows / laptop, CPU)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.smoke_test
```

Expected output: connectome built, forward pass shapes verified,
`runs/videos/smoke_test.mp4` written.

To try a short PPO run on the toy connectome:
```powershell
python -m scripts.train --profile smoke
```

## GPU server (Linux, CUDA)

Two paths — Docker (recommended, no host mess) or a venv on bare metal.

### Option A — Docker (recommended)

```bash
docker compose build train
docker compose run --rm smoke                          # sanity check on the GPU
docker compose up train                                # full training run
docker compose up -d tensorboard                       # http://<host>:6006
```

Outputs land under `./runs/` and `./data/` on the host — both are
bind-mounted so nothing is lost if the container is recreated.

### Option B — bare metal

```bash
bash scripts/setup_gpu_server.sh
source .venv/bin/activate
tmux new -s train
python -m scripts.train --profile full
```

### Two-GPU boxes: one for training, one for Blender

If the server has 2+ GPUs, pin each workload to its own card with
`CUDA_VISIBLE_DEVICES` — this works at the driver level, so neither the
training code nor the Blender script needs any changes; each process just
sees "one GPU" and treats it as device 0.

```bash
nvidia-smi -L                          # confirm both GPUs are visible, note their order

# Terminal / tmux pane 1 — training on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --profile full

# Terminal / tmux pane 2 — Blender/Cycles render on GPU 1 (run any time,
# even while training is still going, since it's a separate physical card)
CUDA_VISIBLE_DEVICES=1 python -m scripts.render_morphology \
    --recording runs/morphology/activation.npz --dry-run
```

`CUDA_VISIBLE_DEVICES` is inherited by subprocesses, so it also correctly
scopes the Blender subprocess that `render_morphology.py` launches — no
separate flag needed for that.

## Getting the real connectome

The smoke and full profiles both default to `source: synthetic`. To use
the actual FlyWire data:

1. Log in at https://codex.flywire.ai and download `neurons.csv` and
   `connections.csv` (or run `python -m scripts.download_connectome`).
2. Drop both under `data/connectome/raw/`.
3. In `config/config.yaml`, set `connectome.source: flywire`.
4. Delete `data/connectome/flywire_connectome.npz` if it exists — the
   loader will rebuild the cache from the CSVs on next run.

## Configuration knobs worth knowing

| Field | Effect |
|---|---|
| `profile` | Switches all `smoke`/`full` overrides at once. |
| `connectome.source` | `synthetic` (no download) or `flywire` (real data). |
| `model.sim_steps` | Inner ticks of the recurrent connectome per env step. Higher = more computation per decision. |
| `model.train_synapses` | If false, only the sensory-projection and motor-readout layers learn; synapses are frozen at their biological weights. |
| `telemetry.enabled` | Toggles the 3-panel MP4 recording during training. |
| `env.n_envs` | Parallel envs (per profile). SubprocVecEnv is used when >1. |

## 3D brain telemetry

The center panel is a live, slowly-rotating 3D point-cloud of the connectome
([src/telemetry/brain3d.py](src/telemetry/brain3d.py)) — each neuron is a
point at its anatomical position, brightness-coded by that step's activation.
With `connectome.source: flywire`, positions come from the real soma/nucleus
coordinates in the Codex export (when present — see `_extract_positions` in
[src/connectome/loader.py](src/connectome/loader.py)); with `synthetic`, a
fabricated bilateral two-lobe layout stands in so the viewer still reads as
"a brain." Pure NumPy + OpenCV (vectorised scatter + one Gaussian blur), no
OpenGL/EGL context required — safe under Docker/Xvfb.

## Offline research-grade morphology render (real traced neurons)

The live telemetry panel above is fast but stylized (points, not traced
morphology). For the Janelia/neuVid-style visualization — **real traced
neuron skeletons in the actual CNS shape (optic lobes + central brain +
ventral nerve cord), lighting up frame-by-frame as the agent plays** — use
the offline Blender pipeline in [src/morphology/](src/morphology/) +
[blender/render_activation.py](blender/render_activation.py).

This is a post-training step, not live: full traced morphology can't be
rendered per training step (the research videos are offline renders too).

### Prerequisites

1. Train with `connectome.source: flywire` so neurons carry real MCNS
   bodyIds (synthetic graphs can't map to real skeletons).
2. A free [neuprint](https://neuprint.janelia.org) account + API token:
   ```bash
   export NEUPRINT_TOKEN=<your token from the account page>
   ```
3. Render extras + Blender (Blender is a separate, non-pip install):
   ```bash
   pip install -r requirements-render.txt
   bash scripts/setup_blender.sh          # portable Blender, no root needed
   export BLENDER_BIN=$(cat .blender_bin)
   ```

### Headless GPU boxes (Vast.ai etc.)

Blender runs fine headless — it's a portable binary, no root/apt/GUI. The one
gotcha is the **render engine**:

- **Cycles (default here)** renders fully headless on the GPU via OptiX/CUDA
  with **no display and no Xvfb**. Use this on Vast.ai.
- **EEVEE** is faster but on headless Linux needs **EGL** (Xvfb is *not*
  enough — OpenGL isn't invoked under a virtual display). Only pass
  `--engine BLENDER_EEVEE` if you've set up EGL.

`scripts/setup_blender.sh` downloads a portable Blender and prints a one-liner
to confirm the GPU is visible to Cycles. Always do a `--dry-run` first (renders
a single still) to catch GPU/skeleton/material issues before the full animation.

### Three steps

```bash
# 1. Record the connectome's activation over a rollout (CPU ok, no token).
#    Spotlights motor + sensory + a fill sample, capped by --max-neurons.
python -m scripts.record_activation --profile full \
    --checkpoint runs/checkpoints/ppo_connectome_full.zip \
    --out runs/morphology/activation.npz --steps 900 --max-neurons 400

# 2+3. Fetch real skeletons/neuropil meshes from neuprint, then render in
#      Blender. (Orchestrated; each stage is skippable via --skip-fetch /
#      --skip-render for iteration.)
python -m scripts.render_morphology \
    --recording runs/morphology/activation.npz \
    --out runs/videos/morphology.mp4 --engine BLENDER_EEVEE
```

`--max-neurons` matters: rendering full traced morphology is only feasible
for hundreds of neurons, so the recorder captures a spotlight subset (all
motor + sensory neurons, then a random fill) rather than all ~166k.

Data flow: `record_activation` → `activation.npz` (per-neuron timeseries
keyed to real bodyIds) → `neuprint_fetch` → `skeletons/*.swc` +
`neuropil/*.obj` → Blender keyframes emission per neuron → `morphology.mp4`.

## Architecture in one paragraph

Observations from Flappy Bird's 12-dim state vector are projected onto
`n_sensory` afferent neurons (labelled `sensory`/`visual`/`optic` in the
FlyWire super-class field; on the synthetic graph, cells with low
in-degree). The recurrent network runs `sim_steps` sparse GEMM ticks
over the connectome adjacency matrix. `n_motor` efferent neurons feed
two linear heads: a 2-way action logit (flap/fall) and a scalar value.
PPO trains the synapse weights, the sensory projection, and the readout
heads simultaneously — but the graph topology stays frozen.

## Tests

```bash
python -m pytest -q
```

Covers connectome shape, sensory/motor disjointness, forward-pass
shapes, gradient flow to synapse weights, dashboard composition, and
MP4 write.
