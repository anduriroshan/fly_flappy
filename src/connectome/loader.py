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
    # Real dataset body/root id per neuron index (n_neurons,), int64. This is
    # the bridge from our 0..N-1 RL neuron indices back to real neuprint
    # bodyIds, needed to fetch traced morphology for the offline render.
    # None for synthetic graphs (no real neurons to map to).
    node_ids: Optional[np.ndarray] = None
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
        node_ids=spec.node_ids if spec.node_ids is not None else np.array([], dtype=np.int64),
        source=np.array(spec.source),
    )


def _load_cache(path: Path) -> ConnectomeSpec:
    z = np.load(path, allow_pickle=True)
    labels = z["labels"]
    positions = z["positions"] if "positions" in z else np.array([])
    node_ids = z["node_ids"] if "node_ids" in z else np.array([])
    return ConnectomeSpec(
        n_neurons=int(z["n_neurons"]),
        rows=z["rows"],
        cols=z["cols"],
        weights=z["weights"],
        labels=labels if len(labels) else None,
        positions=positions if positions.size else None,
        node_ids=node_ids if node_ids.size else None,
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

    # Codex/MCNS exports use human-readable "Title Case" headers (e.g. "Root
    # ID", "Super Class"); other releases use snake_case. Try both.
    id_col = _find(neurons_df.columns, ["root_id", "Root ID", "Root id", "id"])
    pre_col = _find(conn_df.columns, ["pre_root_id", "pre_pt_root_id", "pre"])
    post_col = _find(conn_df.columns, ["post_root_id", "post_pt_root_id", "post"])
    syn_col = _find(conn_df.columns, ["syn_count", "n_syn", "weight"])

    all_ids = neurons_df[id_col].to_numpy()

    # Subsample if the full graph exceeds the configured budget. Uniform
    # random node choice thins edges quadratically (both endpoints must
    # survive independently) — at 15k/166k neurons that drops mean synapses
    # per neuron from ~54 to ~5. Snowball sampling grows outward from real
    # synaptic neighbourhoods instead, preserving realistic local density.
    if n_neurons < len(all_ids):
        full_id_to_pos = {int(rid): i for i, rid in enumerate(all_ids)}
        pre_pos_all = conn_df[pre_col].map(full_id_to_pos)
        post_pos_all = conn_df[post_col].map(full_id_to_pos)
        valid = pre_pos_all.notna() & post_pos_all.notna()
        pre_pos_all = pre_pos_all[valid].to_numpy(dtype=np.int64)
        post_pos_all = post_pos_all[valid].to_numpy(dtype=np.int64)
        keep_pos = _snowball_sample(pre_pos_all, post_pos_all, len(all_ids), n_neurons, seed)
        keep_ids = all_ids[keep_pos]
    else:
        keep_ids = all_ids

    id_to_idx = {int(rid): i for i, rid in enumerate(keep_ids)}
    mask = conn_df[pre_col].isin(id_to_idx) & conn_df[post_col].isin(id_to_idx)
    conn_df = conn_df.loc[mask]

    rows = conn_df[pre_col].map(id_to_idx).to_numpy(dtype=np.int64)
    cols = conn_df[post_col].map(id_to_idx).to_numpy(dtype=np.int64)
    weights = conn_df[syn_col].to_numpy(dtype=np.float32)

    # Inhibitory sign — GABAergic/glutamatergic synapses flip negative.
    # Prefer a populated per-connection nt_type column; several MCNS/Codex
    # releases ship this column present but entirely empty (as filtered
    # connection tables do), in which case fall back to the pre-synaptic
    # neuron's *predicted* transmitter — NT identity is a per-neuron property
    # (Dale's principle), so this is a reasonable substitute for a missing
    # per-synapse label, and arguably more correct anyway.
    conn_nt_col = _find(conn_df.columns, ["nt_type", "predicted_nt_type"], required=False)
    inh = None
    if conn_nt_col is not None and conn_df[conn_nt_col].notna().any():
        inh = conn_df[conn_nt_col].isin(["GABA", "GLUT"]).to_numpy()
    else:
        neuron_nt_col = _find(
            neurons_df.columns,
            ["Predicted NT type", "predicted_nt_type", "nt_type"],
            required=False,
        )
        if neuron_nt_col is not None:
            nt_by_id = neurons_df.set_index(id_col)[neuron_nt_col]
            pre_nt = conn_df[pre_col].map(nt_by_id)
            inh = pre_nt.isin(["GABA", "GLUT"]).to_numpy()
    if inh is not None:
        weights = np.where(inh, -weights, weights)

    labels = neurons_df.set_index(id_col).reindex(keep_ids)
    label_col = _find(labels.columns, ["super_class", "Super Class", "Super class"], required=False)
    label_col = label_col or labels.columns[0]
    label_arr = labels[label_col].fillna("unknown").to_numpy(dtype=object)

    positions = _extract_positions(labels)

    return ConnectomeSpec(
        n_neurons=len(keep_ids),
        rows=rows,
        cols=cols,
        weights=weights,
        labels=label_arr,
        positions=positions,
        node_ids=np.asarray(keep_ids, dtype=np.int64),
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


def _snowball_sample(pre_pos: np.ndarray, post_pos: np.ndarray, n_total: int,
                     n_target: int, seed: int) -> np.ndarray:
    """Pick `n_target` node positions by growing outward from random seed
    neurons along real synaptic edges, instead of scattering independent
    random picks. Restarts from a fresh random seed whenever the current
    component is exhausted, so disconnected components don't stall growth.
    """
    ones = np.ones(pre_pos.size, dtype=np.int8)
    adj = sparse.coo_matrix((ones, (pre_pos, post_pos)), shape=(n_total, n_total)).tocsr()
    adj = adj.maximum(adj.T)  # treat as undirected for traversal purposes

    rng = np.random.default_rng(seed)
    visited = np.zeros(n_total, dtype=bool)
    order: list[int] = []

    while len(order) < n_target:
        remaining = np.where(~visited)[0]
        if remaining.size == 0:
            break
        seed_node = int(rng.choice(remaining))
        visited[seed_node] = True
        order.append(seed_node)
        frontier = np.array([seed_node])

        while frontier.size and len(order) < n_target:
            neighbor_idx = np.unique(adj[frontier].indices)
            neighbor_idx = neighbor_idx[~visited[neighbor_idx]]
            if neighbor_idx.size == 0:
                break
            budget_left = n_target - len(order)
            take = neighbor_idx[:budget_left]
            visited[take] = True
            order.extend(take.tolist())
            frontier = take

    return np.array(order[:n_target], dtype=np.int64)


def _find(cols, candidates, required: bool = True):
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(f"None of {candidates} found in {list(cols)}")
    return None
