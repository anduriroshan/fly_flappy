"""Offline research-grade morphology rendering pipeline.

This package produces the Janelia/neuVid-style visualization: real traced
neuron skeletons rendered in Blender, lit up frame-by-frame according to
the connectome policy's activation during a Flappy Bird rollout.

Pipeline (see scripts/render_morphology.py for the orchestration):

    1. record_activation -> ActivationRecording (.npz): per-neuron
       activation timeseries for a spotlight subset, keyed to real bodyIds.
       Runs anywhere (CPU, no token).
    2. gcs_fetch          -> SWC skeletons, fetched directly from Janelia's
       public GCS bucket (CC-BY licensed, no login/token — same bucket
       Neuroglancer streams from client-side). This is the default/primary
       path; only needs `requests`, already a core dependency. Verified
       working end-to-end 2026-09-13.
       neuprint_fetch     -> alternative path via navis + neuprint-python,
       for neuropil ROI meshes (not available in the public skeleton
       bucket) or if you specifically want neuprint's own dataset access.
       Needs NEUPRINT_TOKEN + `pip install -r requirements-render.txt`.
    3. blender render      -> MP4. Needs Blender (+ ideally a GPU) on the
       server. Driven by blender/render_activation.py.

Steps 1-2 (gcs_fetch variant) need nothing beyond the base requirements.
neuprint_fetch imports navis lazily so it doesn't affect the rest of the
package if those extras aren't installed.
"""
from .recorder import ActivationRecording, record_activation
from .gcs_fetch import fetch_skeletons_swc
