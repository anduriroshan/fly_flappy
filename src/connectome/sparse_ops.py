"""Memory-safe sparse synaptic propagation with a custom autograd backward.

Why this exists
---------------
The connectome update is `drive[b, i] = sum_j W[i, j] * v[b, j]`, i.e.
`(W @ v.T).T`, where W is the fixed-topology sparse synapse matrix whose
~6.2M nonzero *values* are the learnable weights.

`torch.sparse.mm`'s builtin backward w.r.t. those sparse values materializes
a DENSE n x n gradient before masking it back to the sparse pattern. At the
full MCNS scale (n = 166,700) that is a 166700^2 * 4 B = 111 GB allocation
and simply OOMs — training the real connectome was impossible.

The gradient is actually sparse: for edge e = (i, j),
    grad_value[e] = sum_b grad_drive[b, i] * v[b, j]
which touches only the nnz real synapses, never the n x n dense grid. This
module computes exactly that (chunked over the batch to bound the transient
(chunk x nnz) buffer), removing the neuron ceiling entirely.

Forward still uses `torch.sparse.mm` — it is already memory-efficient; only
the values-gradient was the problem.
"""
from __future__ import annotations

import os

import torch


# Cap on the transient (chunk x nnz) buffer used in the values-gradient, in
# number of elements. This was originally set tiny (50M ~ 200MB) purely to be
# safe before we had real GPU headroom numbers -- but a too-small budget means
# more, smaller sequential chunks, i.e. more GPU kernel launches doing less
# work each, which costs real wall-clock time without saving any memory that
# was actually needed. E.g. at the full MCNS scale (nnz ~ 6.2M), the old 50M
# budget forced a chunk size of 8, meaning a batch of 256 needed 32 sequential
# passes through the backward loop instead of 1.
#
# Default here (2B elements ~ 8GB at fp32) comfortably fits a 24GB card with
# room to spare, but is NOT auto-sized to whatever GPU happens to be present.
# Override via FLY_FLAPPY_GRAD_ELEM_BUDGET if you hit an OOM on a smaller
# card, or want to push it higher on a bigger one.
_GRAD_ELEM_BUDGET = int(os.environ.get("FLY_FLAPPY_GRAD_ELEM_BUDGET", 2_000_000_000))


class _SparseSynapseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, v, n):
        # drive[b, row] = sum over edges of values * v[b, col]
        # W[row, col] = values  (row = post-synaptic, col = pre-synaptic)
        W = torch.sparse_coo_tensor(indices, values, (n, n),
                                    is_coalesced=True, check_invariants=False)
        drive = torch.sparse.mm(W, v.t()).t().contiguous()
        ctx.save_for_backward(values, indices, v)
        ctx.n = n
        return drive

    @staticmethod
    def backward(ctx, grad_drive):
        values, indices, v = ctx.saved_tensors
        row, col = indices[0], indices[1]
        grad_drive = grad_drive.contiguous()
        grad_values = grad_v = None

        # --- gradient w.r.t. the sparse VALUES (the learnable synapses) ---
        # grad_value[e] = sum_b grad_drive[b, row[e]] * v[b, col[e]]
        # Computed at the nnz real edges only — never a dense n x n matrix.
        if ctx.needs_input_grad[0]:
            nnz = values.shape[0]
            B = grad_drive.shape[0]
            chunk = max(1, _GRAD_ELEM_BUDGET // max(nnz, 1))
            grad_values = torch.zeros_like(values)
            for s in range(0, B, chunk):
                gd = grad_drive[s:s + chunk]          # (c, n)
                vv = v[s:s + chunk]                    # (c, n)
                grad_values += (gd[:, row] * vv[:, col]).sum(dim=0)

        # --- gradient w.r.t. the input activations v ---
        # grad_v = (W^T @ grad_drive.T).T ; sparse-dense mm, stays memory-safe.
        if ctx.needs_input_grad[2]:
            Wt = torch.sparse_coo_tensor(
                torch.stack([col, row]), values, (ctx.n, ctx.n),
                check_invariants=False,
            )
            grad_v = torch.sparse.mm(Wt, grad_drive.t()).t()

        return grad_values, None, grad_v, None


def sparse_synapse_drive(values: torch.Tensor, indices: torch.Tensor,
                         v: torch.Tensor, n: int) -> torch.Tensor:
    """Compute `(W @ v.T).T` for W[row,col]=values, with a sparse backward.

    values  : (nnz,) synaptic weights (may require grad)
    indices : (2, nnz) coalesced [row=post, col=pre] indices
    v       : (B, n) presynaptic activations
    n       : number of neurons

    Dispatch: for n up to ~50k the builtin autograd path through
    torch.sparse.mm is used — dense n^2 gradient allocation is n^2 * 4B
    (2.5GB at n=25k, 10GB at n=50k), well within a 24GB GPU. Only above
    that does the memory savings of the custom op matter, so it's kept
    for future scale-up to the full ~166k MCNS budget (dense would be
    111 GB there — see the module docstring). The custom path did have
    a numerically-silent correctness bug on some torch+CUDA
    combinations that took a lot of RL-hours to catch; the builtin path
    is well-tested by torch itself, so at the sizes it can handle,
    prefer it.
    """
    if n <= 50_000:
        W = torch.sparse_coo_tensor(indices, values, (n, n),
                                    is_coalesced=True, check_invariants=False)
        return torch.sparse.mm(W, v.t()).t().contiguous()
    return _SparseSynapseMM.apply(values, indices, v, n)
