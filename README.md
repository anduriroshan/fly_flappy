# fly-flappy

Drive a Flappy Bird RL agent with the adult *Drosophila melanogaster*
connectome (MCNS v1.0, ~166,700 real neurons, ~50M synapses). PPO
optimises **real synaptic weights** over the fixed biological topology;
the agent's flap/fall decision emerges from motor neurons downstream of
sensory neurons that receive the game state.

Two things to run:

- **[Web platform](server/)** — a local browser demo where a trained fly
  plays Flappy Bird live, with real traced neuron morphology lighting up
  from actual per-neuron activation. CPU-only, no GPU needed. This is the
  primary deliverable.
- **[GPU training](src/rl/train.py)** — trains a fresh checkpoint on a
  rented GPU box. Produces a single artifact: `runs/checkpoints/*.zip`,
  which is everything the web platform needs.

## Layout

```
fly_flappy/
├── config/config.yaml         # single source of truth for both profiles
├── src/
│   ├── connectome/            # loader, mapping, PyTorch sparse net + custom autograd
│   ├── env/                   # gym wrapper + vec-env factory
│   ├── rl/                    # SB3 policy + training loop
│   ├── morphology/            # public-bucket GCS skeleton fetcher
│   └── utils/                 # device/config helpers
├── server/                    # FastAPI backend for the web platform
├── web/                       # Three.js frontend
├── scripts/                   # entrypoints: train / serve / smoke_test / prepare_brain / download
├── tests/                     # pytest suites
├── requirements.txt           # CPU-local (laptop): training + web platform
├── requirements-gpu.txt       # CUDA 12.1 (GPU server): training only
└── requirements-web.txt       # FastAPI + uvicorn (layer on top of requirements.txt)
```

## Web platform — run the trained fly locally (no GPU)

Assuming you already have a trained checkpoint at
`runs/checkpoints/ppo_connectome_full.zip`:

```bash
pip install -r requirements.txt -r requirements-web.txt
python -m scripts.serve
```

Then open **http://127.0.0.1:8000** and hit **Play**. First run fetches
~2,000 real traced neuron skeletons from Janelia's public MCNS GCS bucket
(no auth) and caches them; subsequent runs start instantly.

See [BRAIN_VISUALIZATION_FAQ.md](BRAIN_VISUALIZATION_FAQ.md) for what's
real vs stylized in the brain panel, why activity clusters where it does,
and known limitations.

## Local sanity check (CPU laptop)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.smoke_test        # synthetic 2k-neuron connectome, ~5s
```

Then a short PPO run on the toy connectome:
```powershell
python -m scripts.train --profile smoke
```

## GPU server training (Linux, CUDA)

See [RUNBOOK.md](RUNBOOK.md) for the phase-by-phase deployment recipe on
a rented Vast.ai box (~2 hours for ~1.2M timesteps at the current
20,000-neuron budget).

Short version:
```bash
bash scripts/setup_gpu_server.sh
source .venv/bin/activate
tmux new -s train
python -m scripts.train --profile full
```

Output: `runs/checkpoints/ppo_connectome_full.zip` — copy this back to
the laptop (`scp`) and the web platform is ready to serve it.

## Getting the real connectome (for training)

The smoke and full profiles default to `sources.smoke: synthetic` /
`sources.full: flywire`. To feed real data into training:

1. Log in at https://codex.flywire.ai and download `neurons.csv` and
   `connections.csv` (or run `python -m scripts.download_connectome`).
2. Drop both under `data/connectome/raw/`.
3. Confirm `config/config.yaml` has `connectome.sources.full: flywire`.
4. Delete `data/connectome/full_flywire_*.npz` if you're changing the
   neuron count — the loader will rebuild the cache on next run.

The web platform doesn't need these CSVs — the trained checkpoint bakes
in real bodyIds, positions and the sensory/motor index sets it needs.

## Configuration knobs worth knowing

| Field | Effect |
|---|---|
| `profile` | Switches all `smoke`/`full` overrides at once. |
| `connectome.sources.<profile>` | `synthetic` (no download) or `flywire` (real data). |
| `connectome.budgets.full.neurons` | How many real neurons are actually trained (snowball-sampled from the ~166K full graph). |
| `model.sim_steps` | Inner ticks of the recurrent connectome per env step. |
| `model.train_synapses` | If false, only the sensory-projection and motor-readout layers learn; synapses frozen at biological weights. |
| `env.n_envs` | Parallel envs (per profile). SubprocVecEnv is used when >1. |

## Architecture in one paragraph

Observations from Flappy Bird's 12-dim state vector are projected onto
`n_sensory` afferent neurons (labelled `sensory`/`visual`/`optic` in the
FlyWire super-class field). The recurrent network runs `sim_steps` sparse
GEMM ticks over the real connectome adjacency matrix. `n_motor` efferent
neurons feed two linear heads: a 2-way action logit (flap/fall) and a
scalar value. PPO trains the real synapse weights + sensory projection +
readout heads simultaneously — but the graph topology stays frozen.
Training at the full-neuron scale is only feasible via a custom sparse
autograd op ([src/connectome/sparse_ops.py](src/connectome/sparse_ops.py))
whose backward computes gradients only at the real nnz edges — the
builtin `torch.sparse.mm` backward would allocate a 111GB dense n^2
gradient and OOM.

## Tests

```bash
python -m pytest -q
```

Covers connectome shape, sensory/motor disjointness, forward-pass
shapes, gradient flow through the sparse synapse matrix (custom
autograd), and the public-bucket skeleton fetcher.
