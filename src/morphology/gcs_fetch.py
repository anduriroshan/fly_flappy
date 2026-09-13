"""Fetch real traced skeletons directly from Janelia's public GCS bucket.

Bypasses neuprint's API/auth entirely. The MCNS dataset's precomputed
segmentation (including per-neuron skeletons) is published as a public,
CC-BY-licensed Google Cloud Storage bucket — the same one Neuroglancer
streams from client-side with no login, which is only possible because it
allows anonymous reads. Verified directly (2026-09-13): a plain HTTPS GET
for a real bodyId returns a valid neuroglancer-precomputed-skeleton binary
with no Authorization header at all.

Bucket root: gs://flyem-male-cns/v1.0/  (equivalently
https://storage.googleapis.com/flyem-male-cns/v1.0/)

Skeleton path used here:
    v1.0/segmentation/skeletons-malecns/skeletons-precomputed/{body_id}

This dataset's `info` is `{"@type": "neuroglancer_skeletons"}` with no
`vertex_attributes` or `transform` — vertex coordinates are plain float32
XYZ in nanometers, used as-is (verified: coordinate magnitudes are ~1e5,
consistent with a ~250-500 micron fly CNS in nm).

Binary format (little-endian), per the neuroglancer precomputed skeleton
spec (https://github.com/google/neuroglancer/.../skeletons.md):
    uint32 num_vertices
    uint32 num_edges
    float32[num_vertices, 3]   vertex positions (x, y, z)
    uint32[num_edges, 2]       edges (vertex index pairs)
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

BUCKET_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation"
SKELETON_PATH = "skeletons-malecns/skeletons-precomputed"


def fetch_skeleton_raw(body_id: int, timeout: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Fetch and parse one neuron's skeleton. Returns (vertices, edges):
    vertices (N, 3) float32 XYZ in nm; edges (E, 2) int64 vertex-index pairs.
    Raises requests.HTTPError if this body_id has no skeleton (e.g. a
    fragment too small to skeletonize)."""
    url = f"{BUCKET_BASE}/{SKELETON_PATH}/{int(body_id)}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return _parse_precomputed_skeleton(resp.content)


def _parse_precomputed_skeleton(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    num_vertices, num_edges = struct.unpack_from("<II", data, 0)
    offset = 8
    vertices = np.frombuffer(data, dtype="<f4", count=num_vertices * 3,
                             offset=offset).reshape(num_vertices, 3).copy()
    offset += num_vertices * 3 * 4
    edges = np.frombuffer(data, dtype="<u4", count=num_edges * 2,
                          offset=offset).reshape(num_edges, 2).astype(np.int64)
    return vertices, edges


def _edges_to_parents(num_vertices: int, edges: np.ndarray) -> np.ndarray:
    """Turn an undirected edge list into a rooted-forest parent array via
    BFS (handles multiple disconnected components — common for skeletons
    with small disconnected fragments — by rooting each separately)."""
    adjacency: list[list[int]] = [[] for _ in range(num_vertices)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    parents = np.full(num_vertices, -1, dtype=np.int64)
    visited = np.zeros(num_vertices, dtype=bool)
    for root in range(num_vertices):
        if visited[root]:
            continue
        visited[root] = True
        frontier = [root]
        while frontier:
            nxt = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        parents[neighbor] = node
                        nxt.append(neighbor)
            frontier = nxt
    return parents


def write_swc(body_id: int, vertices: np.ndarray, edges: np.ndarray, out_path: Path,
             default_radius: float = 50.0) -> None:
    """Write a standard SWC file. Radius is a placeholder (this dataset
    doesn't publish per-vertex radius) — the Blender renderer uses a fixed
    tube radius from its own CLI arg, not the SWC radius column, so this
    value is cosmetic/unused downstream."""
    parents = _edges_to_parents(vertices.shape[0], edges)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write(f"# neuron {body_id}, fetched from public MCNS GCS bucket (no auth)\n")
        for i, (x, y, z) in enumerate(vertices):
            node_id = i + 1
            parent_id = -1 if parents[i] < 0 else int(parents[i]) + 1
            fh.write(f"{node_id} 0 {x:.2f} {y:.2f} {z:.2f} {default_radius:.1f} {parent_id}\n")


def fetch_skeletons_swc(body_ids: Iterable[int], out_dir: str | Path,
                        max_workers: int = 24) -> dict[int, Path]:
    """Public-bucket equivalent of neuprint_fetch.fetch_skeletons_swc — same
    signature/return convention, no token required. Skips bodyIds that
    already have a cached SWC, and logs (without raising) any bodyId with
    no published skeleton rather than aborting the whole batch.

    Fetches concurrently (thread pool) — this is a latency-bound HTTP GET
    per neuron (~870ms measured, mostly network round-trip, not payload
    size), so sequential fetching of a few thousand neurons takes tens of
    minutes for no reason; concurrent requests to the same public bucket
    cut that down substantially. Each worker writes to its own output file
    (no shared state to race on) and any single failure is isolated.
    """
    import concurrent.futures

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[int, Path] = {}
    body_ids = [int(b) for b in body_ids if int(b) >= 0]

    todo = []
    for bid in body_ids:
        p = out_dir / f"{bid}.swc"
        if p.exists():
            written[bid] = p
        else:
            todo.append(bid)

    def _fetch_one(bid: int):
        p = out_dir / f"{bid}.swc"
        vertices, edges = fetch_skeleton_raw(bid)
        write_swc(bid, vertices, edges, p)
        return bid, p

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, bid): bid for bid in todo}
        for fut in concurrent.futures.as_completed(futures):
            bid = futures[fut]
            try:
                bid, p = fut.result()
                written[bid] = p
            except requests.HTTPError as e:
                print(f"[gcs] bodyId {bid}: no published skeleton ({e.response.status_code}), skipping")
            except Exception as e:
                print(f"[gcs] bodyId {bid}: fetch/parse failed ({e}), skipping")
            done += 1
            if done % 100 == 0:
                print(f"[gcs] {done}/{len(todo)} processed, {len(written)} written so far")

    print(f"[gcs] wrote {len(written)}/{len(body_ids)} skeletons -> {out_dir} (public bucket, no auth)")
    return written
