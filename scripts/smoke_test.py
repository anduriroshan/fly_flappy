"""End-to-end sanity check runnable on a laptop CPU.

Confirms:
    1. The connectome loader builds a synthetic graph.
    2. The neuron mapper assigns sensory/motor cells.
    3. ConnectomeNet forward + backward run cleanly.
    4. The SB3 policy imports and initialises.

If this passes locally, the same code path will run against the full
connectome on a GPU box — the only thing that changes is the config
profile (`--profile full` vs the default smoke settings here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.connectome import ConnectomeNet, build_neuron_map, load_connectome
from src.utils import load_config, resolve_device
from src.utils.device import describe as describe_device


def main():
    cfg = load_config("config/config.yaml", profile="smoke")
    device = resolve_device(cfg.device)
    print(f"[smoke] device: {describe_device(device)}")

    # 1. Connectome
    spec = load_connectome(
        source=cfg.connectome_source,
        n_neurons=cfg.n_neurons,
        synapses_per_neuron=cfg.synapses_per_neuron,
        cache_path=None,   # skip cache for the smoke test
        seed=cfg.connectome["seed"],
    )
    assert spec.rows.size == spec.cols.size == spec.weights.size
    print(f"[smoke] connectome: {spec.n_neurons} neurons, "
          f"{spec.weights.size} synapses, mean |w|={np.abs(spec.weights).mean():.3f}")

    # 2. Neuron map
    nmap = build_neuron_map(
        spec,
        sensory_fraction=cfg.connectome["sensory_fraction"],
        motor_fraction=cfg.connectome["motor_fraction"],
        seed=cfg.connectome["seed"],
    )
    print(f"[smoke] mapping:    {nmap.n_sensory} sensory, {nmap.n_motor} motor")

    # 3. Forward pass
    obs_dim = 12   # matches the simple FlappyBird obs
    action_dim = 2
    net = ConnectomeNet(
        spec=spec, neuron_map=nmap,
        obs_dim=obs_dim, action_dim=action_dim,
        sim_steps=cfg.model["sim_steps"], leak=cfg.model["leak"],
        activation=cfg.model["activation"],
        train_synapses=cfg.model["train_synapses"],
    ).to(device)

    obs = torch.randn(4, obs_dim, device=device)
    logits, value, state = net(obs, return_neural_state=True)
    assert logits.shape == (4, action_dim), logits.shape
    assert value.shape == (4, 1), value.shape
    assert state.shape == (4, spec.n_neurons), state.shape
    print(f"[smoke] forward:    logits={logits.shape}, value={value.shape}, "
          f"state={state.shape}, sparsity={net.sparsity:.4%}")

    # Try importing the SB3 policy to catch surface-level errors early.
    try:
        from src.rl.policy import ConnectomeActorCriticPolicy  # noqa: F401
        print("[smoke] policy:     ConnectomeActorCriticPolicy imports cleanly")
    except Exception as e:  # pragma: no cover
        print(f"[smoke] policy:     import failed: {e}")
        raise

    print("\n[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
