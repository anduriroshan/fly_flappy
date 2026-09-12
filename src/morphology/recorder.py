"""Record a connectome policy's per-neuron activation over a rollout.

Produces an ActivationRecording keyed to real dataset bodyIds, which the
neuprint fetch + Blender render steps consume to light up traced skeletons
in sync with what the "brain" was doing while playing the game.

We record only a *spotlight subset* of neurons (motor + sensory + a fill
sample, capped by `max_neurons`) rather than all ~166k — rendering full
traced morphology is only feasible for hundreds of neurons, and the file
would otherwise be enormous (N x T floats).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ActivationRecording:
    """Per-neuron activation timeseries for a spotlight subset of neurons."""
    activations: np.ndarray      # (T, K) float32 — membrane state per step
    node_ids: np.ndarray         # (K,) int64 — real dataset bodyIds
    positions: np.ndarray        # (K, 3) float32 — anatomical positions
    labels: np.ndarray           # (K,) object — super_class labels
    spotlight_idx: np.ndarray    # (K,) int64 — original connectome indices
    role: np.ndarray             # (K,) object — 'motor' | 'sensory' | 'other'
    actions: np.ndarray          # (T,) int64
    rewards: np.ndarray          # (T,) float32
    dataset: str = "male-cns:v1.0"

    @property
    def n_neurons(self) -> int: return int(self.node_ids.size)
    @property
    def n_steps(self) -> int: return int(self.actions.size)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            activations=self.activations,
            node_ids=self.node_ids,
            positions=self.positions,
            labels=self.labels,
            spotlight_idx=self.spotlight_idx,
            role=self.role,
            actions=self.actions,
            rewards=self.rewards,
            dataset=np.array(self.dataset),
        )
        return path

    @staticmethod
    def load(path: str | Path) -> "ActivationRecording":
        z = np.load(path, allow_pickle=True)
        return ActivationRecording(
            activations=z["activations"], node_ids=z["node_ids"],
            positions=z["positions"], labels=z["labels"],
            spotlight_idx=z["spotlight_idx"], role=z["role"],
            actions=z["actions"], rewards=z["rewards"],
            dataset=str(z["dataset"]),
        )


def select_spotlight(neuron_map, n_neurons: int, max_neurons: int,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Choose which neuron indices to record/render.

    Prioritises all motor + sensory neurons (the interpretable I/O of the
    circuit), then fills the remaining budget with a random sample of the
    rest. Returns (indices, role_labels)."""
    rng = np.random.default_rng(seed)
    motor = np.asarray(neuron_map.motor, dtype=np.int64)
    sensory = np.asarray(neuron_map.sensory, dtype=np.int64)

    io = np.concatenate([motor, sensory])
    if io.size > max_neurons:
        io = io[:max_neurons]

    remaining = max_neurons - io.size
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n_neurons), io)
        fill = rng.choice(pool, min(remaining, pool.size), replace=False)
    else:
        fill = np.array([], dtype=np.int64)

    idx = np.concatenate([io, fill]).astype(np.int64)
    motor_set, sensory_set = set(motor.tolist()), set(sensory.tolist())
    role = np.array([
        "motor" if i in motor_set else "sensory" if i in sensory_set else "other"
        for i in idx
    ], dtype=object)
    return idx, role


def record_activation(
    model,
    env,
    spec,
    neuron_map,
    n_steps: int = 900,
    max_neurons: int = 400,
    deterministic: bool = True,
    seed: int = 0,
) -> ActivationRecording:
    """Run `model` in `env` for up to `n_steps`, capturing the connectome's
    membrane state for a spotlight subset each step.

    `model` must expose `.policy.connectome` (our ConnectomeActorCriticPolicy)
    and `.predict`. `env` is a single (non-vectorised) gym env whose obs feed
    the policy. Auto-resets on episode end so the recording spans multiple
    lives if needed.
    """
    import torch

    connectome = model.policy.connectome
    device = next(connectome.parameters()).device

    if spec.node_ids is not None:
        node_ids_full = np.asarray(spec.node_ids, dtype=np.int64)
    else:
        node_ids_full = np.full(spec.n_neurons, -1, dtype=np.int64)

    idx, role = select_spotlight(neuron_map, spec.n_neurons, max_neurons, seed)

    acts, actions, rewards = [], [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_steps):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, state = connectome(obs_t, return_neural_state=True)
        acts.append(state[0, idx].detach().cpu().numpy().astype(np.float32))

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        actions.append(int(action))
        rewards.append(float(reward))
        if terminated or truncated:
            obs, _ = env.reset()

    return ActivationRecording(
        activations=np.stack(acts, axis=0),
        node_ids=node_ids_full[idx],
        positions=spec.positions[idx].astype(np.float32),
        labels=np.asarray(spec.labels, dtype=object)[idx] if spec.labels is not None
                else np.array(["unknown"] * idx.size, dtype=object),
        spotlight_idx=idx,
        role=role,
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dataset="male-cns:v1.0" if spec.source == "flywire" else "synthetic",
    )
