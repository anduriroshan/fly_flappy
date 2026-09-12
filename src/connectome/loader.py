"""Load the FlyWire adult *Drosophila* connectome, or a synthetic stand-in.

Real-source path
----------------
The FlyWire consortium publishes the adult female fly brain connectome via
the Codex portal (https://codex.flywire.ai). The relevant CSV exports are:

    - neurons.csv    (root_id, super_class, cell_type, side, ...,
                      optionally pos_x/pos_y/pos_z soma coordinates —
                      used for the 3D brain viewer, see _extract_positions)
    - connections.csv (pre_root_id, post_root_id, syn_count, ...)

Place both under `data/connectome/raw/` and set
`connectome.source: flywire` in config.yaml. The parser is intentionally
tolerant of the Codex column naming conventions in use as of 2025-Q4.

Synthetic fallback
------------------
`source: synthetic` produces a small-world random graph whose degree
distribution matches the FlyWire mean (~500 synapses per neuron in the
published data; here scaled down). Enough structure for smoke tests but
NOT a scientifically valid model — do not draw biological conclusions
from synthetic runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse

from .geometry import fabricate_bilateral_positions


@dataclass
class ConnectomeSpec:
    """Loaded connectome as an in-memory sparse graph."""
    n_neurons: int
    # CSR adjacency in COO components: rows = pre, cols = post, vals = weight.
    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray
    # Optional metadata (region/type labels, one string per neuron).
    labels: Optional[np.ndarray] = None
    # 3D anatomical position per neuron (n_neurons, 3), float32. Real soma/
    # nucleus coordinates when source='flywire' and Codex exports them;
    # otherwise a fabricated bilateral-lobe layout (see geometry.py) so the
    # 3D brain viewer always has something to render.
    positions: Optional[np.ndarray] = None
    source: str = "synthetic"

    def to_scipy(self) -> sparse.csr_matrix:
        m = sparse.coo_matrix(
            (self.weights, (self.rows, self.cols)),
            shape=(self.n_neurons, self.n_neurons),
            dtype=np.float32,
        )
        return m.tocsr()


# =====================================================================
# Public entry point
# =====================================================================
def load_connectome(
    source: str,
    n_neurons: int,
    synapses_per_neuron: int,
    raw_dir: str | Path = "data/connectome/raw",
    cache_path: str | Path | None = None,
    seed: int = 1337,
) -> ConnectomeSpec:
    """Return a ConnectomeSpec, from cache if available, else build & cache."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            spec = _load_cache(cache_path)
            if spec.positions is None:  # cache predates 3D positions
                spec.positions = fabricate_bilateral_positions(spec.n_neurons, seed)
                _save_cache(spec, cache_path)
            return spec

    if source == "synthetic":
        spec = _build_synthetic(n_neurons, synapses_per_neuron, seed)
    elif source == "flywire":
        spec = _load_flywire(Path(raw_dir), n_neurons, synapses_per_neuron, seed)
    else:
        raise ValueError(f"Unknown connectome source: {source!r}")

    if spec.positions is None:
        # Real data without a parseable position column — fall back so the
        # 3D viewer always has something to render (flagged via source string
        # unchanged; only the position *layout* is fabricated here).
        spec.positions = fabricate_bilateral_positions(spec.n_neurons, seed)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _save_cache(spec, cache_path)
    return spec


# =====================================================================
# Cache
# =====================================================================
def _save_cache(spec: ConnectomeSpec, path: Path) -> None:
    np.savez_compressed(
        path,
        n_neurons=np.int64(spec.n_neurons),
        rows=spec.rows,
        cols=spec.cols,
        weights=spec.weights,
        labels=spec.labels if spec.labels is not None else np.array([], dtype=object),
        positions=spec.positions if spec.positions is not None else np.array([], dtype=np.float32),
        source=np.array(spec.source),
    )


def _load_cache(path: Path) -> ConnectomeSpec:
    z = np.load(path, allow_pickle=True)
    labels = z["labels"]
    positions = z["positions"] if "positions" in z else np.array([])
    return ConnectomeSpec(
        n_neurons=int(z["n_neurons"]),
        rows=z["rows"],
        cols=z["cols"],
        weights=z["weights"],
        labels=labels if len(labels) else None,
        positions=positions if positions.size else None,
        source=str(z["source"]),
    )


# =====================================================================
# Synthetic builder — Watts–Strogatz + heavy-tailed weight jitter.
# =====================================================================
def _build_synthetic(n_neurons: int, k: int, seed: int) -> ConnectomeSpec:
    rng = np.random.default_rng(seed)
    k = min(k, n_neurons - 1)

    # Ring lattice base: each neuron connects to k nearest neighbours on a ring.
    base_rows = np.repeat(np.arange(n_neurons), k)
    offsets = np.tile(np.arange(1, k + 1), n_neurons)
    base_cols = (base_rows + offsets) % n_neurons

    # Rewire ~10% of edges to a random target (small-world property).
    rewire_mask = rng.random(base_cols.shape) < 0.10
    base_cols[rewire_mask] = rng.integers(0, n_neurons, rewire_mask.sum())

    # Log-normal synapse strengths, sign flipped for ~30% inhibitory synapses
    # (roughly matches published GABA fractions in the fly optic lobe).
    weights = rng.lognormal(mean=0.0, sigma=0.5, size=base_cols.shape).astype(np.float32)
    inhibitory = rng.random(weights.shape) < 0.30
    weights[inhibitory] *= -1.0

    labels = np.array([f"synthetic_{i}" for i in range(n_neurons)], dtype=object)
    positions = fabricate_bilateral_positions(n_neurons, seed)
    return ConnectomeSpec(
        n_neurons=n_neurons,
        rows=base_rows.astype(np.int64),
        cols=base_cols.astype(np.int64),
        weights=weights,
        labels=labels,
        positions=positions,
        source="synthetic",
    )


# =====================================================================
# FlyWire loader — tolerant of Codex column naming.
# =====================================================================
def _load_flywire(raw_dir: Path, n_neurons: int, k: int, seed: int) -> ConnectomeSpec:
    """Parse Codex CSV exports. Subsamples to `n_neurons` if requested.

    Expects (in raw_dir):
        neurons.csv     with a root_id column
        connections.csv with (pre_root_id, post_root_id, syn_count)
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "Loading real FlyWire data requires pandas: pip install pandas"
        ) from e

    neurons_csv = raw_dir / "neurons.csv"
    conn_csv = raw_dir / "connections.csv"
    if not (neurons_csv.exists() and conn_csv.exists()):
        raise FileNotFoundError(
            f"Expected {neurons_csv} and {conn_csv}. "
            f"Download from https://codex.flywire.ai."
        )

    neurons_df = pd.read_csv(neurons_csv)
    conn_df = pd.read_csv(conn_csv)

    id_col = "root_id" if "root_id" in neurons_df.columns else neurons_df.columns[0]
    pre_col = _find(conn_df.columns, ["pre_root_id", "pre_pt_root_id", "pre"])
    post_col = _find(conn_df.columns, ["post_root_id", "post_pt_root_id", "post"])
    syn_col = _find(conn_df.columns, ["syn_count", "n_syn", "weight"])

    all_ids = neurons_df[id_col].to_numpy()

    # Subsample if the full graph exceeds the configured budget.
    rng = np.random.default_rng(seed)
    if n_neurons < len(all_ids):
        keep = rng.choice(len(all_ids), n_neurons, replace=False)
        keep_ids = all_ids[keep]
    else:
        keep_ids = all_ids

    id_to_idx = {int(rid): i for i, rid in enumerate(keep_ids)}
    mask = conn_df[pre_col].isin(id_to_idx) & conn_df[post_col].isin(id_to_idx)
    conn_df = conn_df.loc[mask]

    rows = conn_df[pre_col].map(id_to_idx).to_numpy(dtype=np.int64)
    cols = conn_df[post_col].map(id_to_idx).to_numpy(dtype=np.int64)
    weights = conn_df[syn_col].to_numpy(dtype=np.float32)

    # Optional inhibitory sign — Codex exports a `nt_type` column with the
    # dominant neurotransmitter. GABA / Glutamate → inhibitory.
    if "nt_type" in conn_df.columns:
        inh = conn_df["nt_type"].isin(["GABA", "GLUT"]).to_numpy()
        weights = np.where(inh, -weights, weights)

    labels = neurons_df.set_index(id_col).reindex(keep_ids)
    label_col = "super_class" if "super_class" in labels.columns else labels.columns[0]
    label_arr = labels[label_col].fillna("unknown").to_numpy(dtype=object)

    positions = _extract_positions(labels)

    return ConnectomeSpec(
        n_neurons=len(keep_ids),
        rows=rows,
        cols=cols,
        weights=weights,
        labels=label_arr,
        positions=positions,
        source="flywire",
    )


def _extract_positions(neurons_df) -> Optional[np.ndarray]:
    """Best-effort extraction of real soma/nucleus 3D coordinates from Codex.

    Returns None (never raises) if no recognisable position columns exist —
    the caller fabricates a placeholder layout in that case. Verify against
    the current Codex export schema; column names have shifted before.
    """
    candidate_sets = [
        ("pos_x", "pos_y", "pos_z"),
        ("nucleus_x", "nucleus_y", "nucleus_z"),
        ("soma_x", "soma_y", "soma_z"),
        ("pt_position_x", "pt_position_y", "pt_position_z"),
        ("x", "y", "z"),
    ]
    for xc, yc, zc in candidate_sets:
        if xc in neurons_df.columns and yc in neurons_df.columns and zc in neurons_df.columns:
            xyz = neurons_df[[xc, yc, zc]].to_numpy(dtype=np.float32)
            if not np.isnan(xyz).all():
                return xyz
    return None


def _find(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of {candidates} found in {list(cols)}")
