"""Monkeypatches _SparseSynapseMM.backward to print what EVERY individual
invocation computes, without modifying sparse_ops.py. Runs the real
ConnectomeNet at sim_steps=2 (the exact failing case) and sim_steps=3 (which
was observed to still come out nonzero overall) so we can see whether one
specific call's own grad_values is silently zero even while grad_drive
flowing into it is not.

Not part of the test suite — delete after the bug is found.
"""
import torch

from src.connectome import sparse_ops

_orig_backward = sparse_ops._SparseSynapseMM.backward
_call_counter = [0]


def _traced_backward(ctx, grad_drive):
    _call_counter[0] += 1
    call_id = _call_counter[0]
    values, indices, v = ctx.saved_tensors
    print(f"  [call {call_id}] needs_input_grad={ctx.needs_input_grad}  "
          f"grad_drive.abs().sum()={grad_drive.abs().sum().item():.6f}  "
          f"v.abs().sum()={v.abs().sum().item():.6f}  "
          f"values.abs().sum()={values.abs().sum().item():.6f}")
    result = _orig_backward(ctx, grad_drive)
    grad_values = result[0]
    print(f"  [call {call_id}] -> grad_values is None? {grad_values is None}  "
          f"abs().sum()={None if grad_values is None else grad_values.abs().sum().item():.6f}")
    return result


sparse_ops._SparseSynapseMM.backward = staticmethod(_traced_backward)

print("torch:", torch.__version__, "cuda:", torch.version.cuda,
      "available:", torch.cuda.is_available())

from src.connectome.loader import load_connectome  # noqa: E402
from src.connectome.mapping import build_neuron_map  # noqa: E402
from src.connectome.model import ConnectomeNet  # noqa: E402

spec = load_connectome("synthetic", n_neurons=64, synapses_per_neuron=4, cache_path=None)
nmap = build_neuron_map(spec, sensory_fraction=0.1, motor_fraction=0.05)
torch.manual_seed(0)
obs = torch.randn(2, 4)

for sim_steps in (2, 3):
    print()
    print(f"=== sim_steps={sim_steps} ===")
    _call_counter[0] = 0
    net = ConnectomeNet(spec, nmap, obs_dim=4, action_dim=2, sim_steps=sim_steps, train_synapses=True)
    logits, value, _ = net(obs)
    (logits.sum() + value.sum()).backward()
    g = net._syn_values.grad
    print(f"FINAL _syn_values.grad.abs().sum() = {g.abs().sum().item():.6f}")
