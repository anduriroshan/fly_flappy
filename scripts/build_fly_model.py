"""Assemble the NeuroMechFly (flygym) simplified STL parts into a single GLB
for the web 'THE FLY' panel.

Source: NeLy-EPFL/flygym, Apache-2.0. The simplified_max2000faces mesh set
(~3.3MB, 39 parts) plus rigging.yaml (per-part pos+quat relative to parent)
and the anatomy tree from flygym/anatomy.py. Each STL is centered at its own
origin, so we accumulate parent transforms down the kinematic tree to place
every part in a shared world frame, bake that into the geometry, mirror the
left side to make the right, and export one named-node GLB. Materials and
animation are applied in the browser (see web/main.js).

Run once (after cloning flygym) to regenerate web/assets/fly/fly.glb:
    FLYGYM_ASSETS=/path/to/flygym/src/flygym/assets/model/neuromechfly \
        python -m scripts.build_fly_model
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "web" / "assets" / "fly"

# flygym asset root (override via env). Defaults to the temp clone location.
FLYGYM = Path(os.environ.get(
    "FLYGYM_ASSETS",
    r"C:/Users/anduri.roshan/AppData/Local/Temp/flygym/src/flygym/assets/model/neuromechfly",
))
MESH_DIR = FLYGYM / "meshes" / "simplified_max2000faces"
RIGGING = FLYGYM / "rigging.yaml"

# ---- anatomy tree, transcribed from flygym/anatomy.py ----
LEG_LINKS = ["coxa", "trochanterfemur", "tibia", *(f"tarsus{s}" for s in "12345")]
ANTENNA_LINKS = ["pedicel", "funiculus", "arista"]
PROBOSCIS_LINKS = ["rostrum", "haustellum"]
ABDOMEN_LINKS = ["abdomen12", *(f"abdomen{s}" for s in "3456")]


def _chain(*names):
    return [(names[i], names[i + 1]) for i in range(len(names) - 1)]


# Build the left+center half only; the right side is mirrored geometrically.
PAIRS = [
    ("c_thorax", "c_head"),
    *_chain("c_head", *(f"c_{lk}" for lk in PROBOSCIS_LINKS)),
    *_chain("c_thorax", *(f"c_{lk}" for lk in ABDOMEN_LINKS)),
    ("c_head", "l_eye"),
    *_chain("c_head", *(f"l_{lk}" for lk in ANTENNA_LINKS)),
    ("c_thorax", "l_wing"),
    ("c_thorax", "l_haltere"),
    *(edge for leg in ("lf", "lm", "lh")
      for edge in _chain("c_thorax", *(f"{leg}_{lk}" for lk in LEG_LINKS))),
]
PARENT = {child: parent for parent, child in PAIRS}
ROOT_SEG = "c_thorax"

# Appendages to mirror left->right (leading 'l' side prefix -> 'r').
MIRROR_PREFIXES = ("l_", "lf_", "lm_", "lh_")

# flygym stores meshes in metres but the rigging positions are in mm; the
# MuJoCo model applies mesh scale [1000,1000,1000] (see mujoco_globals.yaml).
# Without this the parts are ~1000x too small and scatter as specks.
MESH_SCALE = 1000.0


def quat_to_mat(q):
    """MuJoCo quat [w,x,y,z] -> 3x3 rotation."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def local_T(entry):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(entry["quat"])
    T[:3, 3] = entry["pos"]
    return T


def world_T(name, rig, cache):
    if name in cache:
        return cache[name]
    T = local_T(rig[name])
    parent = PARENT.get(name)
    if parent is not None:
        T = world_T(parent, rig, cache) @ T
    cache[name] = T
    return T


def main():
    rig = yaml.safe_load(RIGGING.read_text())
    cache = {}
    scene = trimesh.Scene()
    pivots = {}

    MIRROR = np.diag([1.0, -1.0, 1.0])   # reflect across the sagittal (Y=0) plane

    for name in PARENT.keys() | {ROOT_SEG}:
        stl = MESH_DIR / f"{name}.stl"
        if not stl.exists():
            print(f"  skip {name}: no mesh")
            continue
        mesh = trimesh.load(stl, process=False)
        mesh.apply_scale(MESH_SCALE)      # metres -> mm, matching the rigging
        Tw = world_T(name, rig, cache)
        mesh.apply_transform(Tw)          # bake world transform into vertices
        scene.add_geometry(mesh, node_name=name, geom_name=name)

        # Right-side counterpart: mirror the baked world mesh across Y=0.
        if name.startswith(MIRROR_PREFIXES):
            rname = "r" + name[1:]
            rmesh = mesh.copy()
            rmesh.apply_transform(trimesh.transformations.reflection_matrix(
                [0, 0, 0], [0, 1, 0]))
            rmesh.invert()                # fix winding after reflection
            scene.add_geometry(rmesh, node_name=rname, geom_name=rname)

    # Animation pivots (world hinge points) for the browser.
    for seg in ("l_wing", "lf_coxa"):
        p = world_T(seg, rig, cache)[:3, 3]
        pivots[seg] = p.tolist()
        pivots["r" + seg[1:]] = (MIRROR @ p).tolist()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    glb_path = OUT_DIR / "fly.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))

    bounds = scene.bounds
    meta = {
        "pivots": pivots,
        "bbox_min": bounds[0].tolist(),
        "bbox_max": bounds[1].tolist(),
        "center": ((bounds[0] + bounds[1]) / 2).tolist(),
        "size": (bounds[1] - bounds[0]).tolist(),
    }
    (OUT_DIR / "fly_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[build_fly_model] wrote {glb_path} ({glb_path.stat().st_size/1e6:.2f} MB)")
    print(f"[build_fly_model] nodes: {len(scene.geometry)}  bbox size: {meta['size']}")
    print(f"[build_fly_model] pivots: {list(pivots)}")


if __name__ == "__main__":
    main()
