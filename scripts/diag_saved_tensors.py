"""Test whether the all-zero grad on torch 2.5.1 is caused by stale
saved_tensors: patch the sparse op to CLONE the tensors it saves for
backward, and see if the real ConnectomeNet gradient comes back to life.

If this fixes it -> the underlying bug is torch 2.5.1 keeping a
reference to `v` that gets aliased/overwritten by the recurrent loop's
next-step assignment (v = ... + act_fn(...)) before backward reads it.
Delete after we find the cause.
"""
import torch

from src.connectome import sparse_ops
from src.connectome.sparse_ops import _SparseSynapseMM

print("torch:", torch.__version__, "cuda:", torch.version.cuda,
      "available:", torch.cuda.is_available())


class _ClonedSparseSynapseMM(torch.autograd.Function):
    """Same math as _SparseSynapseMM, but detaches+clones every saved tensor.
    If this fixes the all-zero-grad bug, the original was reading stale
    memory in backward — a torch-2.5-era autograd Function gotcha."""

    @staticmethod
    def forward(ctx, values, indices, v, n):
        W = torch.sparse_coo_tensor(indices, values, (n, n),
                                    is_coalesced=True, check_invariants=False)
        drive = torch.sparse.mm(W, v.t()).t().contiguous()
        # THE ONLY DIFFERENCE FROM sparse_ops.py: .clone() each saved tensor.
        ctx.save_for_backward(values.clone(), indices.clone(), v.clone())
        ctx.n = n
        return drive

    @staticmethod
    def backward(ctx, grad_drive):
        values, indices, v = ctx.saved_tensors
        row, col = indices[0], indices[1]
        grad_drive = grad_drive.contiguous()
        grad_values = grad_v = None

        if ctx.needs_input_grad[0]:
            nnz = values.shape[0]
            B = grad_drive.shape[0]
            chunk = max(1, sparse_ops._GRAD_ELEM_BUDGET // max(nnz, 1))
            grad_values = torch.zeros_like(values)
            for s in range(0, B, chunk):
                gd = grad_drive[s:s + chunk]
                vv = v[s:s + chunk]
                grad_values += (gd[:, row] * vv[:, col]).sum(dim=0)

        if ctx.needs_input_grad[2]:
            Wt = torch.sparse_coo_tensor(
                torch.stack([col, row]), values, (ctx.n, ctx.n),
                check_invariants=False,
            )
            grad_v = torch.sparse.mm(Wt, grad_drive.t()).t()

        return grad_values, None, grad_v, None


def _patched_drive(values, indices, v, n):
    return _ClonedSparseSynapseMM.apply(values, indices, v, n)


# Monkeypatch the module-level function ConnectomeNet imports and calls.
sparse_ops.sparse_synapse_drive = _patched_drive
import src.connectome.model as model_mod   # noqa: E402
model_mod.sparse_synapse_drive = _patched_drive

from src.connectome.loader import load_connectome  # noqa: E402
from src.connectome.mapping import build_neuron_map  # noqa: E402
from src.connectome.model import ConnectomeNet  # noqa: E402

spec = load_connectome("synthetic", n_neurons=64, synapses_per_neuron=4, cache_path=None)
nmap = build_neuron_map(spec, sensory_fraction=0.1, motor_fraction=0.05)

torch.manual_seed(0)
obs = torch.randn(2, 4)

for sim_steps in (2, 3):
    net = ConnectomeNet(spec, nmap, obs_dim=4, action_dim=2, sim_steps=sim_steps, train_synapses=True)
    logits, value, _ = net(obs)
    (logits.sum() + value.sum()).backward()
    g = net._syn_values.grad
    gsum = None if g is None else g.abs().sum().item()
    verdict = "PASS" if (g is not None and gsum > 0) else "FAIL"
    print(f"sim_steps={sim_steps}: FINAL _syn_values.grad.abs().sum() = {gsum:.6f}  -> {verdict}")

# For extra info, also compare against the ORIGINAL (unpatched) op in the
# same process, so we can see side-by-side that patching is what changed
# things (not e.g. a random-seed difference).
print()
print("--- for comparison, ORIGINAL (unpatched) op in the same process ---")
sparse_ops.sparse_synapse_drive = lambda values, indices, v, n: _SparseSynapseMM.apply(values, indices, v, n)
model_mod.sparse_synapse_drive = sparse_ops.sparse_synapse_drive

torch.manual_seed(0)
obs = torch.randn(2, 4)
for sim_steps in (2, 3):
    net = ConnectomeNet(spec, nmap, obs_dim=4, action_dim=2, sim_steps=sim_steps, train_synapses=True)
    logits, value, _ = net(obs)
    (logits.sum() + value.sum()).backward()
    g = net._syn_values.grad
    gsum = None if g is None else g.abs().sum().item()
    print(f"[original op] sim_steps={sim_steps}: FINAL _syn_values.grad.abs().sum() = {gsum:.6f}")
