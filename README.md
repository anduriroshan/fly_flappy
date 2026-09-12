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
