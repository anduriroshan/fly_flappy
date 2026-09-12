"""Unit tests for the connectome + model glue. Run with: python -m pytest -q"""
import numpy as np
import torch

from src.connectome import ConnectomeNet, build_neuron_map, load_connectome


def test_synthetic_shape():
    spec = load_connectome("synthetic", n_neurons=256, synapses_per_neuron=8, cache_path=None)
    assert spec.n_neurons == 256
    assert spec.rows.shape == spec.cols.shape == spec.weights.shape
    assert spec.rows.max() < 256


def test_synthetic_positions_populated():
    spec = load_connectome("synthetic", n_neurons=128, synapses_per_neuron=6, cache_path=None)
    assert spec.positions is not None
    assert spec.positions.shape == (128, 3)
    assert spec.positions.dtype == np.float32


def test_mapping_disjoint():
    spec = load_connectome("synthetic", n_neurons=200, synapses_per_neuron=6, cache_path=None)
    nmap = build_neuron_map(spec, sensory_fraction=0.1, motor_fraction=0.05)
    assert len(set(nmap.sensory.tolist()) & set(nmap.motor.tolist())) == 0


def test_forward_pass_shapes():
    spec = load_connectome("synthetic", n_neurons=128, synapses_per_neuron=6, cache_path=None)
    nmap = build_neuron_map(spec, sensory_fraction=0.1, motor_fraction=0.05)
    net = ConnectomeNet(spec, nmap, obs_dim=12, action_dim=2, sim_steps=2)
    obs = torch.randn(3, 12)
    logits, value, state = net(obs, return_neural_state=True)
    assert logits.shape == (3, 2)
    assert value.shape == (3, 1)
    assert state.shape == (3, 128)
    assert net.neuron_positions.shape == (128, 3)


def test_gradient_flows_to_synapses():
    spec = load_connectome("synthetic", n_neurons=64, synapses_per_neuron=4, cache_path=None)
    nmap = build_neuron_map(spec, sensory_fraction=0.1, motor_fraction=0.05)
    net = ConnectomeNet(spec, nmap, obs_dim=4, action_dim=2, sim_steps=2, train_synapses=True)
    logits, value, _ = net(torch.randn(2, 4))
    (logits.sum() + value.sum()).backward()
    assert net._syn_values.grad is not None
    assert net._syn_values.grad.abs().sum() > 0
