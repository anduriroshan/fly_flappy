"""Pick which neurons act as sensory inputs and motor outputs.

Real biology: FlyWire labels neurons with a `super_class` string —
'sensory', 'ascending', 'central', 'motor', etc. When those labels are
present we honour them. When they aren't (synthetic graphs), we fall back
to picking by degree: high in-degree cells become motor neurons (they
integrate lots of upstream signal), and cells with the lowest in-degree
become sensory (they are near the periphery).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .loader import ConnectomeSpec


@dataclass
class NeuronMap:
    """Which indices carry env observations in, and which drive the action."""
    sensory: np.ndarray   # shape (n_sensory,) — indices into the connectome
    motor:   np.ndarray   # shape (n_motor,)
    n_neurons: int

    @property
    def n_sensory(self) -> int: return int(self.sensory.size)
    @property
    def n_motor(self) -> int: return int(self.motor.size)


def build_neuron_map(
    spec: ConnectomeSpec,
    sensory_fraction: float,
    motor_fraction: float,
    seed: int = 0,
) -> NeuronMap:
    n = spec.n_neurons
    n_sensory = max(1, int(sensory_fraction * n))
    n_motor = max(1, int(motor_fraction * n))

    labels = spec.labels
    if labels is not None and any(_is_biological(str(l)) for l in labels[:100]):
        sensory, motor = _pick_by_labels(labels, n_sensory, n_motor, seed)
    else:
        sensory, motor = _pick_by_degree(spec, n_sensory, n_motor, seed)

    # Guarantee disjointness — a neuron can't be both eye and wing muscle.
    motor = np.setdiff1d(motor, sensory, assume_unique=False)
    if motor.size < n_motor:
        # Backfill motors from unused indices.
        used = np.union1d(sensory, motor)
        pool = np.setdiff1d(np.arange(n), used)
        rng = np.random.default_rng(seed + 1)
        extra = rng.choice(pool, n_motor - motor.size, replace=False)
        motor = np.concatenate([motor, extra])

    return NeuronMap(sensory=sensory.astype(np.int64),
                     motor=motor.astype(np.int64),
                     n_neurons=n)


def _is_biological(label: str) -> bool:
    label = label.lower()
    return any(key in label for key in ("sensory", "motor", "ascending", "visual", "optic"))


def _pick_by_labels(labels: np.ndarray, n_sensory: int, n_motor: int, seed: int):
    rng = np.random.default_rng(seed)
    lower = np.array([str(x).lower() for x in labels])

    sensory_mask = np.array([("sensory" in l or "visual" in l or "optic" in l) for l in lower])
    motor_mask = np.array([("motor" in l or "descending" in l) for l in lower])

    sensory_pool = np.where(sensory_mask)[0]
    motor_pool = np.where(motor_mask)[0]

    if sensory_pool.size < n_sensory:
        pad = rng.choice(np.setdiff1d(np.arange(labels.size), sensory_pool),
                         n_sensory - sensory_pool.size, replace=False)
        sensory_pool = np.concatenate([sensory_pool, pad])
    if motor_pool.size < n_motor:
        pad = rng.choice(np.setdiff1d(np.arange(labels.size), motor_pool),
                         n_motor - motor_pool.size, replace=False)
        motor_pool = np.concatenate([motor_pool, pad])

    sensory = rng.choice(sensory_pool, n_sensory, replace=False)
    motor = rng.choice(motor_pool, n_motor, replace=False)
    return sensory, motor


def _pick_by_degree(spec: ConnectomeSpec, n_sensory: int, n_motor: int, seed: int):
    n = spec.n_neurons
    in_deg = np.bincount(spec.cols, minlength=n)
    out_deg = np.bincount(spec.rows, minlength=n)

    # Sensory = low in-degree (few upstream signals) but non-trivial out-degree.
    sensory_score = out_deg - in_deg
    sensory = np.argsort(-sensory_score)[:n_sensory]

    # Motor = high in-degree (they integrate lots of upstream signals).
    motor_score = in_deg - out_deg
    motor = np.argsort(-motor_score)[:n_motor]
    return sensory, motor
