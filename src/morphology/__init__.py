"""Offline research-grade morphology rendering pipeline.

This package produces the Janelia/neuVid-style visualization: real traced
neuron skeletons (fetched from the MCNS `male-cns:v1.0` dataset via navis +
neuprint) rendered in Blender, lit up frame-by-frame according to the
connectome policy's activation during a Flappy Bird rollout.

Pipeline (see scripts/render_morphology.py for the orchestration):

    1. record_activation  -> ActivationRecording (.npz): per-neuron
       activation timeseries for a spotlight subset, keyed to real bodyIds.
       Runs anywhere (CPU, no token).
    2. neuprint_fetch     -> SWC skeletons + neuropil ROI meshes for those
       bodyIds. Needs a neuprint API token (NEUPRINT_TOKEN env var).
    3. blender render      -> MP4. Needs Blender (+ ideally a GPU) on the
       server. Driven by blender/render_activation.py.

Only step 1 is importable without the render extras installed; steps 2-3
import navis / run under Blender and are invoked from their scripts.
"""
from .recorder import ActivationRecording, record_activation
