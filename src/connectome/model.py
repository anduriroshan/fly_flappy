"""ConnectomeNet: a sparse recurrent network wired by a biological graph.

Simulation model per env step:
    1. Encode observation → analog current injected into sensory neurons.
    2. For `sim_steps` inner ticks:
         v ← (1 − leak) · v + activation(W · v + input)
       where W is the fixed-topology sparse synaptic matrix. The weights
       W are learnable if `train_synapses` is True; the *topology* is
       always frozen — the biology stays intact.
    3. Read out motor-neuron activations → project to action logits & value.

The sparse propagation goes through `sparse_ops.sparse_synapse_drive`, a
custom autograd op whose backward computes the synapse-weight gradient only
at the real edges — the builtin torch.sparse.mm backward would allocate a
dense n x n gradient and OOM at the full ~166k-neuron MCNS scale. On CPU
laptops with the smoke budget (~2k neurons), the same code path runs
unmodified.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .loader import ConnectomeSpec
from .mapping import NeuronMap
from .sparse_ops import sparse_synapse_drive


class ConnectomeNet(nn.Module):
    def __init__(
        self,
        spec: ConnectomeSpec,
        neuron_map: NeuronMap,
        obs_dim: int,
        action_dim: int,
        sim_steps: int = 4,
        leak: float = 0.15,
        activation: str = "tanh",
        train_synapses: bool = True,
    ) -> None:
        super().__init__()
        self.n_neurons = spec.n_neurons
        self.sim_steps = int(sim_steps)
        self.leak = float(leak)
        self.act_fn = _activation(activation)

        # ---- Sparse synapse matrix ----
        idx = torch.from_numpy(np.stack([spec.cols, spec.rows], axis=0)).long()
        vals = torch.from_numpy(spec.weights).float()
        # Coalesce ONCE here — merges any duplicate (pre, post) pairs (real
        # data can have the same pair connected across multiple neuropils)
        # and sorts indices into canonical order. Without this, forward()
        # was re-running .coalesce() over the full edge list on every single
        # call (~45% of forward cost at full scale) even though the
        # topology never changes after construction.
        coalesced = torch.sparse_coo_tensor(
            idx, vals, size=(self.n_neurons, self.n_neurons), check_invariants=False
        ).coalesce()
        self._syn_indices = nn.Parameter(coalesced.indices(), requires_grad=False)
        self._syn_values = nn.Parameter(coalesced.values(), requires_grad=train_synapses)

        # ---- Sensory input projection ----
        # A learnable MLP that expands the obs vector to `n_sensory` currents.
        self.sensory_idx = nn.Parameter(
            torch.from_numpy(neuron_map.sensory).long(), requires_grad=False
        )
        self.sensory_proj = nn.Linear(obs_dim, neuron_map.n_sensory)

        # ---- Motor readout ----
        self.motor_idx = nn.Parameter(
            torch.from_numpy(neuron_map.motor).long(), requires_grad=False
        )
        # Two heads: action logits and value.
        self.action_head = nn.Linear(neuron_map.n_motor, action_dim)
        self.value_head = nn.Linear(neuron_map.n_motor, 1)

        # Membrane state, allocated lazily on the first forward pass so that
        # we know the correct device/dtype.
        self.register_buffer("_v_template", torch.zeros(self.n_neurons), persistent=False)

        # 3D anatomical position per neuron — always populated by the loader
        # (real soma coords or a fabricated bilateral-lobe layout). Consumed
        # by the telemetry dashboard's 3D brain renderer.
        self.register_buffer(
            "neuron_positions", torch.from_numpy(spec.positions).float(), persistent=False
        )

        # Real dataset bodyId per neuron index (or an all -1 sentinel for
        # synthetic graphs). Persisted so a reloaded checkpoint can map its
        # RL neuron indices back to real neuprint neurons for the offline
        # morphology render.
        node_ids = spec.node_ids if spec.node_ids is not None else np.full(self.n_neurons, -1, dtype=np.int64)
        self.register_buffer(
            "neuron_ids", torch.from_numpy(np.asarray(node_ids, dtype=np.int64)), persistent=True
        )

    # ---------- properties ----------
    @property
    def sparsity(self) -> float:
        nnz = self._syn_values.numel()
        return 1.0 - nnz / (self.n_neurons * self.n_neurons)

    @property
    def has_real_ids(self) -> bool:
        return bool((self.neuron_ids >= 0).any())

    def get_synapse_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pre_idx, post_idx, weight) numpy arrays — same convention
        as ConnectomeSpec.rows/cols/weights. Lets telemetry code (which only
        has the reloaded policy, not the original spec) recover the edge
        list for the synaptic-pathway visualization."""
        idx = self._syn_indices.detach().cpu().numpy()
        post, pre = idx[0], idx[1]
        weights = self._syn_values.detach().cpu().numpy()
        return pre, post, weights

    # ---------- forward ----------
    def forward(
        self,
        obs: torch.Tensor,
        return_neural_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        obs : (B, obs_dim)
        Returns
        -------
        action_logits : (B, action_dim)
        value         : (B, 1)
        neural_state  : (B, n_neurons) or None
        """
        # Gym envs commonly hand back float64 observations; our weights are
        # float32, and torch.nn.Linear requires matching dtypes.
        obs = obs.to(dtype=self.sensory_proj.weight.dtype)
        B = obs.shape[0]
        device = obs.device

        # Build the per-batch input current. Scatter the projected obs
        # vector into the `n_sensory` slots of a zeroed neuron vector.
        input_current = torch.zeros(B, self.n_neurons, device=device, dtype=obs.dtype)
        sensory_signal = self.sensory_proj(obs)                       # (B, n_sensory)
        input_current.index_add_(1, self.sensory_idx, sensory_signal)

        v = torch.zeros(B, self.n_neurons, device=device, dtype=obs.dtype)
        for _ in range(self.sim_steps):
            # Custom sparse op: memory-safe backward w.r.t. synapse weights
            # (the builtin torch.sparse.mm backward would OOM on a dense n x n
            # gradient at full connectome scale). See sparse_ops.py.
            drive = sparse_synapse_drive(
                self._syn_values, self._syn_indices, v, self.n_neurons
            )
            v = (1.0 - self.leak) * v + self.act_fn(drive + input_current)

        motor_activity = v.index_select(1, self.motor_idx)            # (B, n_motor)
        logits = self.action_head(motor_activity)
        value = self.value_head(motor_activity)
        return logits, value, (v if return_neural_state else None)


def _activation(name: str):
    name = name.lower()
    return {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}[name]
