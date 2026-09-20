"""Sparse synaptic propagation via scatter_add.

Why this exists
---------------
The connectome update is `drive[b, i] = sum_j W[i, j] * v[b, j]`, i.e.
`(W @ v.T).T`, where W is the fixed-topology sparse synapse matrix whose
nonzero *values* are the learnable weights.

torch.sparse.mm looks like the obvious primitive, but its autograd on
recent torch builds (2.5+ observed across CPU and CUDA) silently returns
zero gradients w.r.t. the learnable sparse values when this specific
usage pattern is chained through more than one call on the same leaf
parameter — which is exactly what the recurrent sim_steps loop does.
The failure is not detected by the op itself, PPO just silently trains
only the linear read-in/read-out heads and leaves the connectome frozen.
So we avoid torch.sparse.* here entirely.

Implementation
--------------
Direct scatter-add over the edge list, using only well-tested autograd
primitives (index_select, elementwise multiply, scatter_add). The
required intermediate is `(B, nnz)`, not `n x n` — so this is actually
lighter than the naive torch.sparse.mm backward path (which materializes
a dense n x n) and comparable to the previous hand-rolled custom
autograd Function, without the platform brittleness.

Memory at real training scale (n=30k, nnz≈3.6M, batch=256, sim_steps=4):
each call retains ~3.7 GB for backward; four chained calls ~15 GB. Fits
comfortably on a 24 GB card alongside PPO buffers and activations.
"""
from __future__ import annotations

import torch


def sparse_synapse_drive(values: torch.Tensor, indices: torch.Tensor,
                         v: torch.Tensor, n: int) -> torch.Tensor:
    """Compute `(W @ v.T).T` for W[row,col]=values via scatter_add.

    values  : (nnz,) synaptic weights (may require grad)
    indices : (2, nnz) [row=post, col=pre] indices, coalesced (deduped)
    v       : (B, n) presynaptic activations
    n       : number of neurons
    """
    row, col = indices[0], indices[1]
    B = v.shape[0]
    gathered = v.index_select(1, col)          # (B, nnz)  pre-synaptic activations per edge
    weighted = gathered * values.unsqueeze(0)  # (B, nnz)  per-edge current contribution
    drive = torch.zeros(B, n, device=v.device, dtype=v.dtype)
    row_idx = row.unsqueeze(0).expand(B, -1)   # (B, nnz)
    return drive.scatter_add(1, row_idx, weighted)
