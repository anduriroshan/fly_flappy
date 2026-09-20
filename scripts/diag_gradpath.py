"""Definitive check: is the zero synapse-gradient a broken op, or a fragile
test whose tiny graph has no direct sensory->motor edge within 2 hops?

Prints, on THIS machine:
  1. The n=64 test graph's sensory/motor sets and how many DIRECT
     sensory->motor edges exist (structural gradient path count).
  2. scatter_add forward vs a dense reference matmul (correctness of the op
     itself, independent of the network).
  3. Synapse gradient at a REALISTIC scale (n=2048, sim_steps=4) -- what
     actually matters for training.

Not part of the test suite -- delete once understood.
"""
import numpy as np
import torch

from src.connectome.loader import load_connectome
from src.connectome.mapping import build_neuron_map
from src.connectome.model import ConnectomeNet
from src.connectome.sparse_ops import sparse_synapse_drive

print("torch:", torch.__version__, " numpy:", np.__version__,
      " cuda:", torch.cuda.is_available())

# ---- 1. Structural gradient path in the n=64 test graph ----
print("\n=== 1. n=64 test graph: direct sensory->motor edges ===")
spec = load_connectome("synthetic", n_neurons=64, synapses_per_neuron=4, cache_path=None)
nmap = build_neuron_map(spec, sensory_fraction=0.1, motor_fraction=0.05)
sensory = set(nmap.sensory.tolist())
motor = set(nmap.motor.tolist())
print("sensory idx:", sorted(sensory))
print("motor idx:  ", sorted(motor))
# edges: pre=cols, post=rows  (W[row,col]=weight, drive[row]=sum_col w*v[col])
pre = spec.cols
post = spec.rows
direct = [(int(p), int(q)) for p, q in zip(pre, post) if int(p) in sensory and int(q) in motor]
print(f"direct sensory->motor edges (pre in sensory, post in motor): {len(direct)}")
print("  ", direct[:20])
print("=> with sim_steps=2 + zero-init v, ONLY these edges can get gradient.")
print("=> if this is 0, a zero synapse-grad is CORRECT, not a bug.")

# ---- 2. Op correctness: scatter_add vs dense reference ----
# Coalesce first (dedupe (row,col) pairs), exactly like model.py does before
# it ever calls the op -- otherwise a dense reference that ASSIGNS
# W[row,col]=value would overwrite duplicate edges while scatter_add
# (correctly) SUMS them, giving a spurious mismatch that is an artifact of
# the reference, not the op.
print("\n=== 2. scatter_add op vs dense reference (op correctness) ===")
torch.manual_seed(0)
n = 64
raw_idx = torch.from_numpy(np.stack([spec.rows, spec.cols], axis=0)).long()
raw_vals = torch.randn(raw_idx.shape[1])
co = torch.sparse_coo_tensor(raw_idx, raw_vals, (n, n)).coalesce()
indices = co.indices()
nnz = indices.shape[1]
values = co.values().clone().requires_grad_(True)
v = torch.randn(3, n, requires_grad=True)

drive = sparse_synapse_drive(values, indices, v, n)
# dense reference: build W[row,col]=values, drive = (W @ v.T).T
W = torch.zeros(n, n)
W[indices[0], indices[1]] = values.detach()
ref = (W @ v.detach().t()).t()
print("forward max abs diff vs dense ref:", (drive.detach() - ref).abs().max().item())

drive.sum().backward()
print("values.grad nonzero:", (values.grad != 0).sum().item(), "/", nnz,
      " abs().sum():", values.grad.abs().sum().item())
print("v.grad nonzero:", (v.grad != 0).sum().item(), "/", v.numel())
print("=> if forward diff ~0 and grads nonzero, the op is CORRECT.")

# ---- 3. Realistic scale: does training actually get synapse gradients? ----
print("\n=== 3. realistic scale (n=2048, sim_steps=4): the thing that matters ===")
spec2 = load_connectome("synthetic", n_neurons=2048, synapses_per_neuron=32, cache_path=None)
nmap2 = build_neuron_map(spec2, sensory_fraction=0.05, motor_fraction=0.02)
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = ConnectomeNet(spec2, nmap2, obs_dim=12, action_dim=2, sim_steps=4,
                    train_synapses=True).to(dev)
obs = torch.randn(16, 12, device=dev)
logits, value, _ = net(obs)
(logits.sum() + value.sum()).backward()
g = net._syn_values.grad
print(f"synapse grad nonzero: {(g != 0).sum().item()} / {g.numel()}"
      f"  abs().sum()={g.abs().sum().item():.4f}")
print("=> THIS is what determines whether the connectome learns during training.")
print("=> nonzero here = training will work, regardless of the n=64 test.")
