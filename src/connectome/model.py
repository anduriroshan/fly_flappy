"""ConnectomeNet: a sparse recurrent network wired by a biological graph.

Simulation model per env step:
    1. Encode observation → analog current injected into sensory neurons.
    2. For `sim_steps` inner ticks:
         v ← (1 − leak) · v + activation(W · v + input)
       where W is the fixed-topology sparse synaptic matrix. The weights
       W are learnable if `train_synapses` is True; the *topology* is
       always frozen — the biology stays intact.
    3. Read out motor-neuron activations → project to action logits & value.

Sparse GEMM (torch.sparse.mm) keeps the 139k-neuron matrix tractable on a
single GPU. On CPU laptops with the smoke budget (~2k neurons), the same
code path runs unmodified.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .loader import ConnectomeSpec
from .mapping import NeuronMap


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
        # Torch requires sorted indices for coalesce()
        self._syn_indices = nn.Parameter(idx, requires_grad=False)
        self._syn_values = nn.Parameter(vals, requires_grad=train_synapses)

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

    # ---------- properties ----------
    @property
    def sparsity(self) -> float:
        nnz = self._syn_values.numel()
        return 1.0 - nnz / (self.n_neurons * self.n_neurons)

    def _syn_matrix(self) -> torch.Tensor:
        # We built _syn_indices from arange() in constructor order, so the
        # tensor is safe; disable the invariant check to silence the warning
        # and skip its non-trivial CPU cost on every forward.
        return torch.sparse_coo_tensor(
            self._syn_indices, self._syn_values,
            size=(self.n_neurons, self.n_neurons),
            check_invariants=False,
        ).coalesce()

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
        B = obs.shape[0]
        device = obs.device
        W = self._syn_matrix()

        # Build the per-batch input current. Scatter the projected obs
        # vector into the `n_sensory` slots of a zeroed neuron vector.
        input_current = torch.zeros(B, self.n_neurons, device=device, dtype=obs.dtype)
        sensory_signal = self.sensory_proj(obs)                       # (B, n_sensory)
        input_current.index_add_(1, self.sensory_idx, sensory_signal)

        v = torch.zeros(B, self.n_neurons, device=device, dtype=obs.dtype)
        for _ in range(self.sim_steps):
            # torch.sparse.mm expects (N, K) dense — transpose to (n_neurons, B),
            # multiply, transpose back.
            drive = torch.sparse.mm(W, v.t()).t()
            v = (1.0 - self.leak) * v + self.act_fn(drive + input_current)

        motor_activity = v.index_select(1, self.motor_idx)            # (B, n_motor)
        logits = self.action_head(motor_activity)
        value = self.value_head(motor_activity)
        return logits, value, (v if return_neural_state else None)


def _activation(name: str):
    name = name.lower()
    return {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}[name]
