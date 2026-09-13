"""Blender background script: render real traced neurons lighting up by
connectome activation, neuVid/Janelia style.

Run (on the GPU server, Blender 3.x/4.x installed):

    blender --background --python blender/render_activation.py -- \
        --recording runs/morphology/activation.npz \
        --swc-dir   runs/morphology/skeletons \
        --neuropil-dir runs/morphology/neuropil \
        --out       runs/videos/morphology.mp4 \
        --engine BLENDER_EEVEE --samples 32 --fps 30

This script only uses Blender's bundled Python + numpy (Blender ships numpy).
It does NOT import our src package. Inputs are the artifacts produced by
scripts/record_activation.py (the .npz) and src/morphology/neuprint_fetch.py
(the SWC + OBJ files).

Design:
  - Each neuron's SWC becomes a thin beveled curve (tube) with an emission
    material. Per-neuron emission strength is keyframed from the activation
    timeseries, so neurons brighten/dim as the agent plays.
  - Colour encodes role: motor = warm orange, sensory = cyan, other = violet.
  - Neuropil OBJ meshes become a faint translucent grey backdrop (the CNS
    silhouette: brain + optic lobes + ventral nerve cord).
  - EEVEE + bloom gives the glowing look cheaply; Cycles is available via
    --engine CYCLES for final-quality renders.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import bpy
    import mathutils
except ImportError:
    print("This script must be run inside Blender: blender --background --python ...")
    sys.exit(1)


# ----------------------------- args --------------------------------------
def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--swc-dir", required=True)
    ap.add_argument("--neuropil-dir", default=None)
    ap.add_argument("--out", default="runs/videos/morphology.mp4")
    # CYCLES is the default because it renders fully headless on a GPU box
    # (Vast.ai etc.) with no display/EGL fuss. EEVEE is faster but needs EGL
    # on headless Linux (Xvfb is NOT enough — GL isn't invoked under it).
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Use every Nth recorded step as a rendered frame.")
    ap.add_argument("--emission-gain", type=float, default=8.0)
    ap.add_argument("--base-glow", type=float, default=0.25)
    ap.add_argument("--tube-radius", type=float, default=0.06)
    ap.add_argument("--dry-run", action="store_true",
                    help="Render only the middle frame as a PNG, to validate "
                         "setup (GPU, skeletons, materials) before the full run.")
    ap.add_argument("--gpu-index", type=int, default=None,
                    help="Which GPU device (by index within Blender's own GPU "
                         "device list, NOT necessarily nvidia-smi order) to use. "
                         "CUDA_VISIBLE_DEVICES does not restrict this list, so on "
                         "a multi-GPU box shared with another process (e.g. "
                         "training), pick explicitly rather than trusting the "
                         "default (last device in the list).")
    return ap.parse_args(argv)


ROLE_COLORS = {
    "motor":   (1.0, 0.45, 0.12, 1.0),   # warm orange
    "sensory": (0.15, 0.75, 1.0, 1.0),   # cyan
    "other":   (0.65, 0.35, 1.0, 1.0),   # violet
}


# --------------------------- scene setup ---------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0, 0, 0, 1)   # black world
    return scene


def configure_render(scene, args):
    scene.render.engine = args.engine
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.fps = args.fps
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.filepath = str(Path(args.out))

    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        _enable_gpu(scene, gpu_index=args.gpu_index)
    else:
        # EEVEE path (needs EGL on headless). Its native bloom is a bonus on
        # top of the compositor glare we add below.
        eevee = getattr(scene, "eevee", None)
        if eevee is not None:
            if hasattr(eevee, "use_bloom"):
                eevee.use_bloom = True
            if hasattr(eevee, "taa_render_samples"):
                eevee.taa_render_samples = args.samples

    add_glow_compositor(scene, threshold=args.base_glow)


def add_glow_compositor(scene, threshold: float = 0.25):
    """Engine-agnostic glow via a compositor Glare (fog-glow) node, so the
    emissive neurons bloom even under Cycles (which has no EEVEE-style bloom).
    """
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    glare = nt.nodes.new("CompositorNodeGlare")
    # FOG_GLOW exists across Blender 3.x/4.x; newer builds also have "BLOOM".
    try:
        glare.glare_type = "FOG_GLOW"
    except TypeError:
        pass
    if hasattr(glare, "quality"):
        glare.quality = "HIGH"
    if hasattr(glare, "threshold"):
        glare.threshold = threshold
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], glare.inputs["Image"])
    nt.links.new(glare.outputs["Image"], comp.inputs["Image"])


def _enable_gpu(scene, gpu_index: int | None = None):
    """Enable exactly ONE GPU device for Cycles -- never all matching ones.

    CUDA_VISIBLE_DEVICES does NOT reliably restrict Blender's own OptiX/CUDA
    device enumeration (confirmed: Blender still lists every physical GPU on
    the box regardless of that env var). So on a multi-GPU box where another
    process (e.g. RL training) owns a different physical GPU, enabling every
    device of the chosen backend type would contend with it for VRAM/compute.
    This selects one specific device by index within the GPU-only list
    instead of trusting env-var filtering.
    """
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        chosen_backend = None
        gpu_devices = []
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                gpu_devices = [d for d in prefs.devices if d.type == backend]
                if gpu_devices:
                    chosen_backend = backend
                    break
            except Exception:
                continue
        if chosen_backend is None:
            print("[blender] no CUDA/OptiX GPU found; rendering on CPU")
            scene.cycles.device = "CPU"
            return

        idx = gpu_index if gpu_index is not None else len(gpu_devices) - 1
        idx = max(0, min(idx, len(gpu_devices) - 1))
        target = gpu_devices[idx]

        for d in prefs.devices:
            d.use = (d is target)   # exactly one device, nothing else
        scene.cycles.device = "GPU"
        print(f"[blender] Cycles device[{idx}] of {len(gpu_devices)} {chosen_backend} "
              f"devices found: {target.name!r} -- verify with nvidia-smi during "
              f"render that ONLY this device's utilization/VRAM changes, not the "
              f"one your training job is using")
    except Exception as e:
        print(f"[blender] GPU enable failed, falling back to CPU: {e}")
        scene.cycles.device = "CPU"


# ------------------------------ SWC --------------------------------------
def load_swc(path):
    """Return (coords Nx3 float, parents N int) from an SWC file."""
    ids, coords, parents = [], [], []
    id_to_row = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            nid = int(parts[0]); x, y, z = map(float, parts[2:5]); par = int(parts[6])
            id_to_row[nid] = len(ids)
            ids.append(nid); coords.append((x, y, z)); parents.append(par)
    coords = np.asarray(coords, dtype=np.float64)
    par_rows = np.array([id_to_row.get(p, -1) for p in parents], dtype=np.int64)
    return coords, par_rows


def swc_to_curve(name, coords, parents, radius, scene_center, scene_scale):
    """Build a beveled poly-curve object from SWC nodes+parent links."""
    coords = (coords - scene_center) / scene_scale
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    # Each parent->child link is a 2-point spline. Grouping into polylines per
    # branch would be tidier, but per-edge splines render identically and keep
    # this parser trivial.
    for row, par in enumerate(parents):
        if par < 0:
            continue
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (*coords[par], 1.0)
        spline.points[1].co = (*coords[row], 1.0)
    curve.bevel_depth = radius
    curve.bevel_resolution = 1
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    return obj


def make_emission_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit.inputs["Color"].default_value = color
    emit.inputs["Strength"].default_value = 1.0
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat, emit


# --------------------------- neuropil backdrop ---------------------------
def load_neuropil(neuropil_dir, scene_center, scene_scale):
    d = Path(neuropil_dir)
    objs = sorted(d.glob("*.obj"))
    if not objs:
        return
    mat = bpy.data.materials.new("neuropil")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.65, 1.0)
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 0.05
    mat.blend_method = "BLEND"
    for obj_path in objs:
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_path))
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=str(obj_path))
        for o in bpy.context.selected_objects:
            o.scale = (1.0 / scene_scale,) * 3
            o.location = tuple(-c / scene_scale for c in scene_center)
            o.data.materials.clear()
            o.data.materials.append(mat)


# ------------------------------ camera -----------------------------------
def add_camera(scene):
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (0.0, -4.0, 1.2)
    cam.rotation_euler = (1.2, 0.0, 0.0)
    scene.camera = cam
    return cam


# ------------------------------- main ------------------------------------
def main():
    args = parse_args()
    rec = np.load(args.recording, allow_pickle=True)
    activations = rec["activations"]          # (T, K)
    node_ids = rec["node_ids"]                # (K,)
    roles = rec["role"]                       # (K,)
    T, K = activations.shape

    # Global robust normalisation of |activation| -> 0..1 for emission.
    a = np.abs(activations)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    norm = np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    swc_dir = Path(args.swc_dir)
    # Establish a common scene center/scale from all available skeleton nodes.
    all_coords = []
    swc_cache = {}
    for bid in node_ids:
        p = swc_dir / f"{int(bid)}.swc"
        if p.exists():
            c, par = load_swc(p)
            swc_cache[int(bid)] = (c, par)
            all_coords.append(c)
    if not all_coords:
        print(f"[blender] no SWC files found in {swc_dir}; run the neuprint fetch first.")
        sys.exit(1)
    stacked = np.concatenate(all_coords, axis=0)
    scene_center = stacked.mean(axis=0)
    scene_scale = np.linalg.norm(stacked - scene_center, axis=1).max() or 1.0

    scene = reset_scene()
    configure_render(scene, args)
    add_camera(scene)
    if args.neuropil_dir:
        load_neuropil(args.neuropil_dir, scene_center, scene_scale)

    # Build one curve + emission material per neuron, keyframe its strength.
    emit_nodes = []
    for k, bid in enumerate(node_ids):
        bid = int(bid)
        if bid not in swc_cache:
            emit_nodes.append(None)
            continue
        coords, parents = swc_cache[bid]
        obj = swc_to_curve(f"n{bid}", coords, parents, args.tube_radius,
                           scene_center, scene_scale)
        color = ROLE_COLORS.get(str(roles[k]), ROLE_COLORS["other"])
        mat, emit = make_emission_material(f"m{bid}", color)
        obj.data.materials.append(mat)
        emit_nodes.append(emit)

    # Dry run: set each neuron's emission to its peak-activation frame and
    # render one still PNG, to validate GPU/skeletons/materials fast before
    # committing to the full animation.
    if args.dry_run:
        peak_t = int(np.argmax(norm.mean(axis=1)))
        for k, emit in enumerate(emit_nodes):
            if emit is not None:
                emit.inputs["Strength"].default_value = float(
                    args.base_glow + norm[peak_t, k] * args.emission_gain)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(Path(args.out).with_suffix("")) + "_dryrun.png"
        print(f"[blender] DRY RUN: rendering 1 frame (step {peak_t}), "
              f"{len(swc_cache)} neurons -> {scene.render.filepath}")
        bpy.ops.render.render(write_still=True)
        print("[blender] dry-run done.")
        return

    # Keyframe emission strength across the rollout.
    frames = list(range(0, T, args.frame_stride))
    scene.frame_start = 1
    scene.frame_end = len(frames)
    for f_idx, t in enumerate(frames, start=1):
        for k, emit in enumerate(emit_nodes):
            if emit is None:
                continue
            strength = args.base_glow + norm[t, k] * args.emission_gain
            emit.inputs["Strength"].default_value = float(strength)
            emit.inputs["Strength"].keyframe_insert("default_value", frame=f_idx)

    print(f"[blender] rendering {len(frames)} frames, {len(swc_cache)} neurons "
          f"-> {args.out}")
    bpy.ops.render.render(animation=True)
    print("[blender] done.")


if __name__ == "__main__":
    main()
