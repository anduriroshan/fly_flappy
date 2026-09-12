"""Fabricate 3D neuron positions when real anatomical coordinates are absent.

A fly brain is bilaterally symmetric: two hemisphere lobes (optic lobes +
central brain either side) joined by a central complex. We mimic that
silhouette with two overlapping ellipsoidal point clouds rather than a
single blob, so the synthetic point-cloud viewer still "reads" as a brain.

This is a visual stand-in only — real anatomy comes from FlyWire soma/
nucleus coordinates via `loader._load_flywire`. Do not use fabricated
positions for any biological claim.
"""
from __future__ import annotations

import numpy as np


def fabricate_bilateral_positions(
    n_neurons: int,
    seed: int = 0,
    lobe_offset: float = 1.35,
    lobe_radii: tuple[float, float, float] = (0.9, 1.15, 0.8),
) -> np.ndarray:
    """Return (n_neurons, 3) float32 positions in two overlapping lobes."""
    rng = np.random.default_rng(seed)

    side = rng.integers(0, 2, size=n_neurons)          # 0 = left, 1 = right
    sign = np.where(side == 0, -1.0, 1.0)

    # Points inside a unit ball (rejection-free via normalized Gaussian * radius^(1/3)).
    g = rng.normal(size=(n_neurons, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
    r = rng.random(n_neurons) ** (1.0 / 3.0)
    unit_ball = g * r[:, None]

    rx, ry, rz = lobe_radii
    pts = unit_ball * np.array([rx, ry, rz])
    pts[:, 0] += sign * lobe_offset

    return pts.astype(np.float32)
