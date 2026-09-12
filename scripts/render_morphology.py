"""Orchestrate the offline morphology render:

    activation recording  ->  neuprint skeleton/mesh fetch  ->  Blender render

Typical use on the GPU server (after training + `export NEUPRINT_TOKEN=...`):

    python -m scripts.render_morphology \
        --recording runs/morphology/activation.npz \
        --out runs/videos/morphology.mp4

Each stage can be skipped so you can iterate on just the render:

    --skip-fetch     reuse already-downloaded SWC/OBJ files
    --skip-render    fetch only, don't invoke Blender

The recording must already exist (produce it with scripts/record_activation.py).
Blender is located via --blender, else the BLENDER_BIN env var, else `blender`
on PATH.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.morphology import ActivationRecording


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="runs/morphology/activation.npz")
    ap.add_argument("--swc-dir", default="runs/morphology/skeletons")
    ap.add_argument("--neuropil-dir", default="runs/morphology/neuropil")
    ap.add_argument("--out", default="runs/videos/morphology.mp4")
    ap.add_argument("--dataset", default="male-cns:v1.0")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-neuropil", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--blender", default=os.environ.get("BLENDER_BIN", "blender"))
    # CYCLES renders headless on a GPU box with no display; EEVEE needs EGL.
    ap.add_argument("--engine", default="CYCLES")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="Render a single frame to fail fast on setup issues.")
    args = ap.parse_args()

    rec_path = Path(args.recording)
    if not rec_path.exists():
        sys.exit(f"Recording not found: {rec_path}. Run scripts/record_activation.py first.")
    rec = ActivationRecording.load(rec_path)
    print(f"[render] recording: {rec.n_neurons} neurons x {rec.n_steps} steps, dataset={rec.dataset}")
    real_ids = rec.node_ids[rec.node_ids >= 0]
    if real_ids.size == 0:
        sys.exit("Recording has no real bodyIds (synthetic connectome) — "
                 "re-record with a flywire-source connectome.")

    # ---- Stage 1: fetch skeletons + neuropil meshes ----
    if not args.skip_fetch:
        from src.morphology.neuprint_fetch import (
            get_client, fetch_skeletons_swc, fetch_neuropil_meshes,
        )
        client = get_client(dataset=args.dataset)
        fetch_skeletons_swc(real_ids.tolist(), args.swc_dir, client=client)
        if not args.skip_neuropil:
            fetch_neuropil_meshes(args.neuropil_dir, client=client)
    else:
        print("[render] --skip-fetch: reusing existing SWC/OBJ files")

    # ---- Stage 2: Blender render ----
    if args.skip_render:
        print("[render] --skip-render: stopping after fetch")
        return

    render_script = Path(__file__).resolve().parents[1] / "blender" / "render_activation.py"
    cmd = [
        args.blender, "--background", "--python", str(render_script), "--",
        "--recording", str(rec_path),
        "--swc-dir", str(args.swc_dir),
        "--out", str(args.out),
        "--engine", args.engine,
        "--fps", str(args.fps),
        "--frame-stride", str(args.frame_stride),
    ]
    if args.dry_run:
        cmd += ["--dry-run"]
    if not args.skip_neuropil and Path(args.neuropil_dir).exists():
        cmd += ["--neuropil-dir", str(args.neuropil_dir)]

    print(f"[render] invoking Blender:\n  {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit(f"Blender not found ({args.blender!r}). Install Blender or set "
                 f"--blender / BLENDER_BIN to its path.")
    print(f"[render] done -> {args.out}")


if __name__ == "__main__":
    main()
