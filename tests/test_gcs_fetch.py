"""Offline tests for the public-bucket skeleton fetcher's binary parsing and
SWC conversion — no network access required (that part is already verified
manually against the real bucket; this locks in the parsing logic)."""
import struct

import numpy as np

from src.morphology.gcs_fetch import (
    _edges_to_parents, _parse_precomputed_skeleton, write_swc,
)


def _make_fake_skeleton_bytes(vertices: np.ndarray, edges: np.ndarray) -> bytes:
    header = struct.pack("<II", vertices.shape[0], edges.shape[0])
    return header + vertices.astype("<f4").tobytes() + edges.astype("<u4").tobytes()


def test_parse_precomputed_skeleton_roundtrip():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    raw = _make_fake_skeleton_bytes(vertices, edges)

    parsed_vertices, parsed_edges = _parse_precomputed_skeleton(raw)
    np.testing.assert_allclose(parsed_vertices, vertices)
    np.testing.assert_array_equal(parsed_edges, edges.astype(np.int64))


def test_edges_to_parents_single_chain():
    # 0-1-2-3 chain: vertex 0 should end up as the root.
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    parents = _edges_to_parents(4, edges)
    assert parents[0] == -1
    assert parents[1] == 0
    assert parents[2] == 1
    assert parents[3] == 2


def test_edges_to_parents_handles_disconnected_components():
    # Two separate chains: 0-1 and 2-3. Both must get their own root.
    edges = np.array([[0, 1], [2, 3]])
    parents = _edges_to_parents(4, edges)
    roots = [i for i in range(4) if parents[i] == -1]
    assert len(roots) == 2
    assert parents[1] == 0
    assert parents[3] == 2


def test_write_swc_has_no_orphan_parents(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.uint32)
    out = tmp_path / "999.swc"
    write_swc(999, vertices, edges, out)

    rows = [l.split() for l in out.read_text().splitlines() if not l.startswith("#")]
    ids = {int(r[0]) for r in rows}
    parent_ids = [int(r[6]) for r in rows]
    assert sum(1 for p in parent_ids if p == -1) == 1
    assert all(p == -1 or p in ids for p in parent_ids)
