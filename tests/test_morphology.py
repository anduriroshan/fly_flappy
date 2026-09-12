"""Tests for the offline-morphology parts that run without a neuprint token
or Blender: spotlight selection + ActivationRecording save/load roundtrip."""
import numpy as np

from src.connectome import build_neuron_map, load_connectome
from src.morphology import ActivationRecording
from src.morphology.recorder import select_spotlight


def test_select_spotlight_prioritises_io_and_caps():
    spec = load_connectome("synthetic", n_neurons=500, synapses_per_neuron=8, cache_path=None)
    nmap = build_neuron_map(spec, sensory_fraction=0.1, motor_fraction=0.05)
    idx, role = select_spotlight(nmap, spec.n_neurons, max_neurons=120, seed=0)
    assert idx.size == 120
    assert idx.size == np.unique(idx).size          # no duplicates
    # Every motor + sensory neuron that fits should be present.
    assert set(nmap.motor.tolist()).issubset(set(idx.tolist()))
    roles = set(role.tolist())
    assert "motor" in roles and "sensory" in roles


def test_select_spotlight_small_budget():
    spec = load_connectome("synthetic", n_neurons=500, synapses_per_neuron=8, cache_path=None)
    nmap = build_neuron_map(spec, sensory_fraction=0.1, motor_fraction=0.05)
    idx, role = select_spotlight(nmap, spec.n_neurons, max_neurons=10, seed=0)
    assert idx.size == 10


def test_recording_roundtrip(tmp_path):
    T, K = 12, 30
    rec = ActivationRecording(
        activations=np.random.randn(T, K).astype(np.float32),
        node_ids=np.arange(K, dtype=np.int64),
        positions=np.random.randn(K, 3).astype(np.float32),
        labels=np.array(["x"] * K, dtype=object),
        spotlight_idx=np.arange(K, dtype=np.int64),
        role=np.array(["motor"] * K, dtype=object),
        actions=np.random.randint(0, 2, T).astype(np.int64),
        rewards=np.random.randn(T).astype(np.float32),
        dataset="male-cns:v1.0",
    )
    p = rec.save(tmp_path / "rec.npz")
    loaded = ActivationRecording.load(p)
    assert loaded.n_neurons == K
    assert loaded.n_steps == T
    assert loaded.dataset == "male-cns:v1.0"
    np.testing.assert_allclose(loaded.activations, rec.activations)
