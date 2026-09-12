"""Fetch real traced skeletons + neuropil meshes from neuprint via navis.

Requires:
    pip install -r requirements-render.txt   (navis, neuprint-python)
    export NEUPRINT_TOKEN=...    # from https://neuprint.janelia.org account page

The MCNS dataset is `male-cns:v1.0` on the neuprint.janelia.org server.

navis imports are done lazily inside functions so the rest of the morphology
package (e.g. the recorder) imports fine on machines without the render
extras installed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

DEFAULT_SERVER = "neuprint.janelia.org"
DEFAULT_DATASET = "male-cns:v1.0"


def get_client(dataset: str = DEFAULT_DATASET, server: str = DEFAULT_SERVER,
               token: Optional[str] = None):
    """Build a neuprint Client. Token comes from the NEUPRINT_TOKEN env var
    unless passed explicitly (never hard-code it)."""
    from neuprint import Client

    token = token or os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError(
            "No neuprint token. Create an account at https://neuprint.janelia.org, "
            "copy your API token from the account page, and set NEUPRINT_TOKEN."
        )
    return Client(server, dataset=dataset, token=token)


def fetch_skeletons_swc(
    body_ids: Iterable[int],
    out_dir: str | Path,
    client=None,
    heal: bool = True,
    batch_size: int = 50,
) -> dict[int, Path]:
    """Fetch traced skeletons for `body_ids` and write one SWC per neuron.

    Skips bodyIds whose SWC already exists (resumable). Neurons without a
    skeleton in the dataset are logged and skipped rather than aborting.
    Returns {body_id: swc_path} for the ones successfully written.
    """
    import navis
    import navis.interfaces.neuprint as neu

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = client or get_client()

    body_ids = [int(b) for b in body_ids if int(b) >= 0]
    written: dict[int, Path] = {}
    pending: list[int] = []
    for b in body_ids:
        p = out_dir / f"{b}.swc"
        if p.exists():
            written[b] = p
        else:
            pending.append(b)

    print(f"[neuprint] {len(written)} skeletons cached, fetching {len(pending)}...")
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        try:
            nl = neu.fetch_skeletons(chunk, heal=heal, client=client)
        except Exception as e:
            print(f"[neuprint]   batch {i//batch_size} failed ({e}); skipping")
            continue
        for neuron in nl:
            bid = int(getattr(neuron, "id", getattr(neuron, "name", -1)))
            p = out_dir / f"{bid}.swc"
            try:
                navis.write_swc(neuron, p)
                written[bid] = p
            except Exception as e:
                print(f"[neuprint]   write {bid} failed ({e})")
        print(f"[neuprint]   {min(i+batch_size, len(pending))}/{len(pending)} processed")

    print(f"[neuprint] wrote {len(written)} skeletons -> {out_dir}")
    return written


def fetch_neuropil_meshes(
    out_dir: str | Path,
    client=None,
    rois: Optional[list[str]] = None,
    max_rois: int = 40,
) -> list[Path]:
    """Fetch neuropil ROI meshes as OBJ — these form the faint grey CNS
    backdrop (the brain + VNC silhouette) the neurons sit inside.

    With `rois=None`, grabs the primary ROIs from the dataset hierarchy.
    """
    import navis
    import navis.interfaces.neuprint as neu

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = client or get_client()

    if rois is None:
        try:
            hierarchy = client.fetch_roi_hierarchy(mark_primary=True, format="dict")
            rois = _collect_primary_rois(hierarchy)[:max_rois]
        except Exception as e:
            print(f"[neuprint] could not fetch ROI hierarchy ({e}); pass rois= explicitly")
            return []

    written: list[Path] = []
    for roi in rois:
        p = out_dir / f"{_safe(roi)}.obj"
        if p.exists():
            written.append(p)
            continue
        try:
            vol = neu.fetch_roi(roi, client=client)  # navis.Volume
            navis.write_mesh(vol, p, filetype="obj")
            written.append(p)
        except Exception as e:
            print(f"[neuprint]   ROI {roi} failed ({e})")
    print(f"[neuprint] wrote {len(written)} neuropil meshes -> {out_dir}")
    return written


def _collect_primary_rois(node, acc=None) -> list[str]:
    acc = acc if acc is not None else []
    if isinstance(node, dict):
        for k, v in node.items():
            if k.endswith("*"):  # primary ROIs are marked with a trailing *
                acc.append(k.rstrip("*"))
            _collect_primary_rois(v, acc)
    return acc


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
