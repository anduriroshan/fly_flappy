"""One-off diagnostic for the all-zero synapse-gradient bug seen on the GPU
box. Run this on the affected machine and paste the full output back.

Not part of the test suite / RUNBOOK — delete after the bug is found.
"""
import torch

print("torch:", torch.__version__, "cuda:", torch.version.cuda,
      "available:", torch.cuda.is_available())

# ---- Part 1: the custom sparse op in total isolation, hand-crafted case ----
# 3 neurons, 2 edges: edge0 = (post=0, pre=1) weight 2.0
#                      edge1 = (post=1, pre=2) weight 3.0
# v = [[1.0, 10.0, 100.0]]  (batch=1)
# drive[0] = W[0,1]*v[1] = 2*10 = 20
# drive[1] = W[1,2]*v[2] = 3*100 = 300
# drive[2] = 0
# loss = drive.sum() -> d(loss)/d(values) = [v[pre_of_edge] for each edge]
#      grad_value[edge0] = v[1] = 10.0
#      grad_value[edge1] = v[2] = 100.0
from src.connectome.sparse_ops import sparse_synapse_drive  # noqa: E402

values = torch.tensor([2.0, 3.0], requires_grad=True)
indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # [row(post), col(pre)]
v = torch.tensor([[1.0, 10.0, 100.0]], requires_grad=True)
drive = sparse_synapse_drive(values, indices, v, 3)
print("drive:", drive.tolist(), "expected: [[20.0, 300.0, 0.0]]")
drive.sum().backward()
print("grad_values:", values.grad.tolist(), "expected: [10.0, 100.0]")
part1_ok = values.grad is not None and torch.allclose(values.grad, torch.tensor([10.0, 100.0]))
print("PART 1 (isolated sparse op):", "PASS" if part1_ok else "FAIL")

# ---- Part 2: same synthetic network+sizes as the failing pytest, but with
# full instrumentation of every intermediate stat per sim step ----
print()
print("=" * 60)
from src.connectome.loader import load_connectome  # noqa: E402
from src.connectome.mapping import build_neuron_map  # noqa: E402
from src.connectome.model import ConnectomeNet  # noqa: E402

torch.manual_seed(0)
spec = load_connectome("synthetic", n_neurons=64, synapses_per_neuron=4, cache_path=None)
nmap = build_neuron_map(spec, sensory_fraction=0.1, motor_fraction=0.05)
net = ConnectomeNet(spec, nmap, obs_dim=4, action_dim=2, sim_steps=2, train_synapses=True)

obs = torch.randn(2, 4)
print("obs:", obs.tolist())

# Manually re-run forward() with per-step printouts (duplicated from model.py
# on purpose, so we can inspect internals without modifying the real file).
obs_c = obs.to(dtype=net.sensory_proj.weight.dtype)
B = obs_c.shape[0]
input_current = torch.zeros(B, net.n_neurons)
sensory_signal = net.sensory_proj(obs_c)
input_current.index_add_(1, net.sensory_idx, sensory_signal)
print("input_current stats: mean=%.6f std=%.6f max_abs=%.6f nonzero=%d"
      % (input_current.mean().item(), input_current.std().item(),
         input_current.abs().max().item(), (input_current != 0).sum().item()))

v = torch.zeros(B, net.n_neurons)
v.requires_grad_(False)
for step in range(net.sim_steps):
    drive = sparse_synapse_drive(net._syn_values, net._syn_indices, v, net.n_neurons)
    print(f"step {step}: drive mean=%.6f std=%.6f max_abs=%.6f"
          % (drive.mean().item(), drive.std().item(), drive.abs().max().item()))
    pre_act = drive + input_current
    print(f"step {step}: pre-activation max_abs=%.6f  (tanh saturates hard past ~10)"
          % pre_act.abs().max().item())
    v = (1.0 - net.leak) * v + net.act_fn(pre_act)
    print(f"step {step}: v mean=%.6f std=%.6f max_abs=%.6f"
          % (v.mean().item(), v.std().item(), v.abs().max().item()))

# ---- Part 3: the actual test, exactly as written, for a final pass/fail ----
print()
print("=" * 60)
logits, value, _ = net(obs)
(logits.sum() + value.sum()).backward()
g = net._syn_values.grad
gsum = None if g is None else g.abs().sum().item()
print("PART 3 (real test replica): grad is None?", g is None, " abs().sum() =", gsum)
print("PART 3:", "PASS" if (g is not None and gsum > 0) else "FAIL")
