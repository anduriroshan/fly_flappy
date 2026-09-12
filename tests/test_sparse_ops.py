"""Correctness tests for the custom sparse synaptic op.

The whole point of sparse_synapse_drive is a backward that matches the math
exactly while avoiding the dense n x n gradient. gradcheck (double precision,
numerical vs analytic Jacobian) is the gold standard for that; we also check
the forward equals the reference dense computation.
"""
import numpy as np
import torch

from src.connectome.sparse_ops import sparse_synapse_drive


def _random_sparse(n, nnz, seed=0):
    rng = np.random.default_rng(seed)
    # Unique (row, col) pairs so the coalesced-values assumption holds.
    pairs = set()
    while len(pairs) < nnz:
        pairs.add((int(rng.integers(0, n)), int(rng.integers(0, n))))
    rc = np.array(sorted(pairs), dtype=np.int64).T   # sorted == coalesced order
    indices = torch.from_numpy(rc)
    return indices


def test_forward_matches_dense_reference():
    n, nnz, B = 8, 20, 3
    indices = _random_sparse(n, nnz, seed=1)
    values = torch.randn(nnz, dtype=torch.double)
    v = torch.randn(B, n, dtype=torch.double)

    drive = sparse_synapse_drive(values, indices, v, n)

    # Reference: dense W[row,col]=values ; drive[b,i] = sum_j W[i,j] v[b,j]
    W = torch.zeros(n, n, dtype=torch.double)
    W[indices[0], indices[1]] = values
    ref = v @ W.t()
    assert torch.allclose(drive, ref, atol=1e-10)


def test_gradcheck_values_and_input():
    n, nnz, B = 6, 14, 2
    indices = _random_sparse(n, nnz, seed=2)
    values = torch.randn(nnz, dtype=torch.double, requires_grad=True)
    v = torch.randn(B, n, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda val, vv: sparse_synapse_drive(val, indices, vv, n),
        (values, v), atol=1e-6,
    )


def test_values_grad_is_sparse_shaped():
    """grad w.r.t. values must be (nnz,), never a dense (n, n)."""
    n, nnz, B = 10, 25, 4
    indices = _random_sparse(n, nnz, seed=3)
    values = torch.randn(nnz, dtype=torch.double, requires_grad=True)
    v = torch.randn(B, n, dtype=torch.double)
    drive = sparse_synapse_drive(values, indices, v, n)
    drive.sum().backward()
    assert values.grad.shape == (nnz,)
